#!/usr/bin/env python3
"""
GBM Clinical Query Interface
Interactive interface for clinicians to query the GBM clinical vector database.
Author: Chetanya Pandey
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

class ClinicalQueryInterface:
    def __init__(self, db_dir: str = "vector_db"):
        """Initialize the clinical query interface."""
        self.db_dir = db_dir
        
        # Connect to existing ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get the medical embeddings collection
        try:
            self.collection = self.chroma_client.get_collection("gbm_clinical_medical_embeddings")
        except:
            # Fallback to original collection
            self.collection = self.chroma_client.get_collection("gbm_clinical_data")
        
        # Initialize the same medical embedding model used for database creation
        print("Loading medical domain embedding model for queries...")
        medical_models = [
            'pritamdeka/S-PubMedBert-MS-MARCO',  # PubMedBERT fine-tuned for retrieval
            'pritamdeka/S-BioBert-snli-multinli-stsb',  # BioBERT fine-tuned for sentence similarity
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',  # PubMedBERT base
            'all-MiniLM-L6-v2'  # Fallback general model
        ]
        
        self.embedding_model = None
        for model_name in medical_models:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                print(f"✅ Loaded query embedding model: {model_name}")
                break
            except Exception as e:
                continue
        
        if self.embedding_model is None:
            print("❌ Failed to load medical embedding model, using ChromaDB default")
        
        # Initialize cross-encoder re-ranker for refined semantic matching
        print("Loading cross-encoder re-ranker...")
        cross_encoder_models = [
            'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',  # BioBERT cross-encoder for medical text
            'cross-encoder/ms-marco-MiniLM-L-6-v2',  # MS MARCO cross-encoder
            'cross-encoder/ms-marco-MiniLM-L-4-v2',  # Smaller MS MARCO model
        ]
        
        self.cross_encoder = None
        for model_name in cross_encoder_models:
            try:
                self.cross_encoder = CrossEncoder(model_name)
                print(f"✅ Loaded cross-encoder re-ranker: {model_name}")
                break
            except Exception as e:
                continue
        
        if self.cross_encoder is None:
            print("❌ Failed to load cross-encoder, using metadata re-ranking only")
        
        print(f"✅ Connected to GBM Clinical Database")
        print(f"Total documents: {self.collection.count()}")
        if self.embedding_model:
            print(f"🧠 Query embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        if self.cross_encoder:
            print(f"Cross-encoder re-ranking enabled for refined semantic matching")
    
    def query_clinical_data(self, query: str, n_results: int = 5, metadata_filters: Dict[str, Any] = None, 
                          drug_filter: str = None, section_filter: str = None) -> Dict[str, Any]:
        """Query the clinical database with expanded clinical terminology and medical embeddings."""
        try:
            # Expand query with clinical synonyms and concepts
            expanded_query = self._expand_clinical_query(query)
            
            # Apply explicit retrieval filters
            if drug_filter or section_filter:
                # Use explicit filters for targeted retrieval
                filtered_results = self._apply_explicit_filters(expanded_query, drug_filter, section_filter, n_results * 3)
            else:
                # Build metadata filters based on query content
                if metadata_filters is None:
                    metadata_filters = self._build_metadata_filters(query)
                
                # Use custom medical embeddings if available
                if self.embedding_model:
                    query_embedding = self.embedding_model.encode([expanded_query])
                    filtered_results = self.collection.query(
                        query_embeddings=query_embedding.tolist(),
                        n_results=n_results * 3,  # Get more results for filtering and re-ranking
                        include=['documents', 'metadatas', 'distances'],
                        where=metadata_filters if metadata_filters else None
                    )
                else:
                    # Fallback to text-based query
                    filtered_results = self.collection.query(
                        query_texts=[expanded_query],
                        n_results=n_results * 3,  # Get more results for filtering and re-ranking
                        include=['documents', 'metadatas', 'distances'],
                        where=metadata_filters if metadata_filters else None
                    )
            
            # Apply post-retrieval drug and section filtering
            post_filtered_results = self._post_filter_results(filtered_results, query, drug_filter, section_filter)
            
            # Re-rank results based on metadata relevance
            metadata_reranked = self._rerank_by_metadata(post_filtered_results, query, n_results * 2)
            
            # Apply cross-encoder re-ranking for final refinement
            final_results = self._cross_encoder_rerank(metadata_reranked, query, n_results)
            
            return {
                'query': query,
                'expanded_query': expanded_query,
                'n_results': n_results,
                'results': final_results,
                'metadata_filters': metadata_filters,
                'drug_filter': drug_filter,
                'section_filter': section_filter,
                'using_medical_embeddings': self.embedding_model is not None,
                'using_cross_encoder': self.cross_encoder is not None
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _expand_clinical_query(self, query: str) -> str:
        """Expand query with clinical synonyms and related concepts."""
        query_lower = query.lower()
        expanded_terms = []
        
        # Clinical terminology mappings
        clinical_expansions = {
            # Drug name synonyms
            'temozolomide': ['temozolomide', 'TMZ', 'temodar', 'temodal'],
            'tmz': ['temozolomide', 'TMZ', 'temodar'],
            'temodar': ['temozolomide', 'TMZ', 'temodar'],
            'bevacizumab': ['bevacizumab', 'avastin', 'anti-VEGF'],
            'avastin': ['bevacizumab', 'avastin', 'anti-VEGF'],
            
            # Dosing concepts
            'dose': ['dose', 'dosing', 'dosage', 'mg/m²', 'mg/kg', 'administration'],
            'dosing': ['dose', 'dosing', 'dosage', 'protocol', 'regimen', 'schedule'],
            'maintenance': ['maintenance', 'adjuvant', 'post-radiation', 'cycles'],
            'concurrent': ['concurrent', 'concomitant', 'simultaneous', 'during radiation'],
            'concomitant': ['concomitant', 'concurrent', 'simultaneous', 'chemoradiation'],
            
            # Clinical actions
            'hold': ['hold', 'withhold', 'stop', 'discontinue', 'pause', 'interrupt'],
            'withhold': ['withhold', 'hold', 'stop', 'discontinue'],
            'stop': ['stop', 'discontinue', 'hold', 'withhold', 'cease'],
            'reduce': ['reduce', 'decrease', 'lower', 'modify', 'adjust'],
            'modify': ['modify', 'adjust', 'change', 'alter', 'reduce'],
            
            # Hematologic toxicity
            'thrombocytopenia': ['thrombocytopenia', 'low platelets', 'decreased platelets', 'platelet count'],
            'neutropenia': ['neutropenia', 'low neutrophils', 'decreased ANC', 'neutrophil count'],
            'anemia': ['anemia', 'low hemoglobin', 'decreased Hgb', 'low Hct'],
            'platelets': ['platelets', 'platelet count', 'thrombocytes', 'PLT'],
            'neutrophils': ['neutrophils', 'ANC', 'absolute neutrophil count', 'neutrophil count'],
            
            # Toxicity terms
            'toxicity': ['toxicity', 'adverse effects', 'side effects', 'complications', 'AE'],
            'adverse': ['adverse effects', 'adverse events', 'toxicity', 'side effects', 'AE'],
            'mortality': ['mortality', 'death', 'fatal', 'lethal', 'treatment-related death'],
            
            # GBM terminology
            'glioblastoma': ['glioblastoma', 'GBM', 'glioblastoma multiforme', 'grade IV glioma'],
            'gbm': ['GBM', 'glioblastoma', 'glioblastoma multiforme'],
            'recurrent': ['recurrent', 'progressive', 'relapsed', 'refractory'],
            'newly diagnosed': ['newly diagnosed', 'initial', 'first-line', 'upfront'],
            
            # Clinical monitoring
            'monitoring': ['monitoring', 'surveillance', 'follow-up', 'assessment', 'evaluation'],
            'laboratory': ['laboratory', 'lab values', 'blood work', 'CBC', 'chemistry'],
            'cbc': ['CBC', 'complete blood count', 'blood count', 'hemogram'],
            
            # Treatment phases
            'chemoradiation': ['chemoradiation', 'chemoradiotherapy', 'CCRT', 'concurrent therapy'],
            'radiotherapy': ['radiotherapy', 'radiation therapy', 'RT', 'XRT'],
            
            # MGMT testing
            'mgmt': ['MGMT', 'O6-methylguanine-DNA methyltransferase', 'methylation status'],
            'methylation': ['methylation', 'MGMT status', 'methylated', 'unmethylated'],
            
            # Performance status
            'kps': ['KPS', 'Karnofsky', 'performance status', 'functional status'],
            'performance': ['performance status', 'KPS', 'functional status', 'ECOG'],
        }
        
        # Process the query
        words = query_lower.split()
        expanded_words = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = word.strip('.,;:!?()[]{}')
            
            if clean_word in clinical_expansions:
                # Add all synonyms for this term
                synonyms = clinical_expansions[clean_word]
                expanded_words.extend(synonyms)
            else:
                expanded_words.append(word)
        
        # Create expanded query
        if len(expanded_words) > len(words):
            # Join expanded terms with spaces
            expanded_query = ' '.join(expanded_words)
        else:
            expanded_query = query
        
        return expanded_query
    
    def _build_metadata_filters(self, query: str) -> Dict[str, Any]:
        """Build ChromaDB metadata filters based on query content."""
        query_lower = query.lower()
        
        # ChromaDB requires a single top-level operator, so we'll prioritize the most specific filter
        # Priority: drug > clinical_topic > other metadata
        
        # Check for specific drug mentions first (highest priority)
        if any(drug in query_lower for drug in ['temozolomide', 'tmz', 'temodar']):
            # Return single drug filter - ChromaDB doesn't support $contains, use $eq with partial match
            return None  # Skip complex filtering for now, rely on re-ranking
        elif any(drug in query_lower for drug in ['bevacizumab', 'avastin']):
            return None  # Skip complex filtering for now, rely on re-ranking
        
        # Filter by clinical topic based on query keywords (second priority)
        topic_keywords = {
            'dosing': ['dose', 'dosing', 'mg/m²', 'mg/kg', 'protocol', 'regimen', 'schedule', 'administration'],
            'toxicity': ['toxicity', 'side effects', 'adverse', 'monitoring', 'laboratory', 'CBC', 'platelets', 'neutrophils'],
            'administration': ['give', 'administer', 'infusion', 'oral', 'IV', 'intravenous', 'timing'],
            'monitoring': ['monitor', 'surveillance', 'follow-up', 'assessment', 'blood work', 'lab values'],
            'contraindications': ['contraindication', 'avoid', 'caution', 'warning', 'precaution'],
            'interactions': ['interaction', 'drug interaction', 'concurrent', 'concomitant']
        }
        
        matching_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                matching_topics.append(topic)
        
        if matching_topics:
            # Use single topic filter or $in for multiple topics
            if len(matching_topics) == 1:
                return {'clinical_topic': {'$eq': matching_topics[0]}}
            else:
                return {'clinical_topic': {'$in': matching_topics}}
        
        # No specific filters found - return None to search all documents
        return None
    
    def _apply_explicit_filters(self, query: str, drug_filter: str, section_filter: str, n_results: int) -> Dict[str, Any]:
        """Apply explicit drug and section filters during retrieval."""
        # Get all results first, then filter
        if self.embedding_model:
            query_embedding = self.embedding_model.encode([query])
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=min(n_results * 2, self.collection.count()),  # Get more for filtering
                include=['documents', 'metadatas', 'distances']
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results * 2, self.collection.count()),
                include=['documents', 'metadatas', 'distances']
            )
        
        return results
    
    def _post_filter_results(self, results: Dict[str, Any], query: str, drug_filter: str, section_filter: str) -> Dict[str, Any]:
        """Apply post-retrieval filtering based on drug mentions and sections."""
        if not results['documents'][0] or (not drug_filter and not section_filter):
            return results
        
        # Detect drug filter from query if not explicitly provided
        if not drug_filter:
            drug_filter = self._detect_drug_from_query(query)
        
        filtered_docs = []
        filtered_metadatas = []
        filtered_distances = []
        filtered_ids = []
        
        # Define section keywords for filtering
        section_keywords = {
            'dose_modifications': ['dose modification', 'dose reduction', 'dose adjustment', 'withhold', 'discontinue', 'reduce dose', 'dose interruption', 'toxicity management'],
            'contraindications': ['contraindication', 'contraindicated', 'should not', 'avoid', 'do not use', 'not recommended'],
            'adverse_effects': ['adverse effects', 'side effects', 'toxicity', 'adverse events', 'safety', 'tolerability'],
            'dosing': ['dosage', 'dose', 'mg/m²', 'mg/kg', 'administration', 'protocol', 'regimen'],
            'monitoring': ['monitoring', 'surveillance', 'laboratory', 'CBC', 'blood count', 'assess', 'evaluation']
        }
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            include_result = True
            
            # Apply drug filter
            if drug_filter:
                drugs_in_metadata = metadata.get('drugs', '').lower()
                drugs_in_content = doc.lower()
                
                # Check if specific drug is mentioned
                if drug_filter.lower() == 'temozolomide':
                    drug_terms = ['temozolomide', 'tmz', 'temodar']
                elif drug_filter.lower() == 'bevacizumab':
                    drug_terms = ['bevacizumab', 'avastin']
                else:
                    drug_terms = [drug_filter.lower()]
                
                drug_found = (any(term in drugs_in_metadata for term in drug_terms) or 
                             any(term in drugs_in_content for term in drug_terms))
                
                if not drug_found:
                    include_result = False
            
            # Apply section filter
            if section_filter and include_result:
                if section_filter.lower() in section_keywords:
                    keywords = section_keywords[section_filter.lower()]
                    section_found = any(keyword.lower() in doc.lower() for keyword in keywords)
                    
                    # Also check document metadata for clinical topic
                    clinical_topic = metadata.get('clinical_topic', '').lower()
                    topic_match = any(keyword.lower() in clinical_topic for keyword in keywords)
                    
                    if not (section_found or topic_match):
                        include_result = False
                else:
                    # Direct section name matching
                    if section_filter.lower() not in doc.lower():
                        include_result = False
            
            if include_result:
                filtered_docs.append(doc)
                filtered_metadatas.append(metadata)
                filtered_distances.append(distance)
                filtered_ids.append(results['ids'][0][i] if results.get('ids') else f"filtered_{i}")
        
        # Reconstruct filtered results
        return {
            'ids': [filtered_ids],
            'documents': [filtered_docs],
            'metadatas': [filtered_metadatas],
            'distances': [filtered_distances]
        }
    
    def _detect_drug_from_query(self, query: str) -> str:
        """Automatically detect drug mentions in query for filtering."""
        query_lower = query.lower()
        
        # Check for drug mentions
        if any(drug in query_lower for drug in ['temozolomide', 'tmz', 'temodar']):
            return 'temozolomide'
        elif any(drug in query_lower for drug in ['bevacizumab', 'avastin']):
            return 'bevacizumab'
        
        return None
    
    def _rerank_by_metadata(self, results: Dict[str, Any], query: str, n_results: int) -> Dict[str, Any]:
        """Re-rank results based on metadata relevance to query."""
        if not results['documents'][0]:
            return results
        
        query_lower = query.lower()
        scored_results = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            # Base score is semantic similarity (1 - distance)
            base_score = 1 - distance
            
            # Metadata boost factors
            metadata_boost = 0
            
            # Boost for clinical topic match
            clinical_topic = metadata.get('clinical_topic', '')
            topic_boosts = {
                'dosing': ['dose', 'dosing', 'mg/m²', 'protocol', 'regimen'],
                'toxicity': ['toxicity', 'side effects', 'adverse', 'monitoring', 'CBC'],
                'administration': ['give', 'administer', 'infusion', 'oral', 'IV'],
                'contraindications': ['contraindication', 'avoid', 'caution', 'warning'],
                'interactions': ['interaction', 'concurrent', 'concomitant']
            }
            
            for topic, keywords in topic_boosts.items():
                if clinical_topic == topic and any(keyword in query_lower for keyword in keywords):
                    metadata_boost += 0.15
                    break
            
            # Boost for specific drug mention
            drugs = metadata.get('drugs', '').lower()
            if ('temozolomide' in drugs or 'tmz' in drugs) and any(term in query_lower for term in ['temozolomide', 'tmz', 'temodar']):
                metadata_boost += 0.1
            elif 'bevacizumab' in drugs and any(term in query_lower for term in ['bevacizumab', 'avastin']):
                metadata_boost += 0.1
            
            # Boost for toxicity grades if query mentions specific toxicity
            toxicity_grades = metadata.get('toxicity_grades', '')
            if toxicity_grades and any(term in query_lower for term in ['grade', 'toxicity', 'adverse']):
                metadata_boost += 0.08
            
            # Boost for laboratory values if query mentions monitoring/labs
            lab_values = metadata.get('laboratory_values', '')
            if lab_values and any(term in query_lower for term in ['lab', 'CBC', 'monitor', 'platelets', 'neutrophils']):
                metadata_boost += 0.08
            
            # Boost for evidence level (FDA approved content gets priority)
            evidence_level = metadata.get('evidence_level', '')
            if 'fda approved' in evidence_level.lower():
                metadata_boost += 0.12
            elif 'clinical trial' in evidence_level.lower():
                metadata_boost += 0.06
            
            # Boost for document type relevance
            doc_type = metadata.get('doc_type', '').lower()
            if 'prescribing information' in doc_type and any(term in query_lower for term in ['dose', 'administration', 'prescribing']):
                metadata_boost += 0.1
            elif 'clinical protocol' in doc_type and any(term in query_lower for term in ['protocol', 'regimen', 'treatment']):
                metadata_boost += 0.1
            
            final_score = base_score + metadata_boost
            scored_results.append((i, final_score, doc, metadata, distance))
        
        # Sort by final score (highest first) and take top n_results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_results[:n_results]
        
        # Reconstruct results format
        reranked_results = {
            'ids': [results['ids'][0][item[0]] for item in top_results],
            'documents': [[item[2] for item in top_results]],
            'metadatas': [[item[3] for item in top_results]],
            'distances': [[item[4] for item in top_results]]
        }
        
        return reranked_results
    
    def _cross_encoder_rerank(self, results: Dict[str, Any], query: str, n_results: int) -> Dict[str, Any]:
        """Apply cross-encoder re-ranking for refined semantic matching."""
        if not self.cross_encoder or not results['documents'][0]:
            # If no cross-encoder available, just truncate to n_results
            return {
                'ids': [results['ids'][0][:n_results]] if results.get('ids') else [[]],
                'documents': [results['documents'][0][:n_results]],
                'metadatas': [results['metadatas'][0][:n_results]],
                'distances': [results['distances'][0][:n_results]]
            }
        
        try:
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = []
            original_data = []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Truncate document for cross-encoder (max 512 tokens typically)
                doc_truncated = doc[:2000]  # Approximate token limit
                query_doc_pairs.append([query, doc_truncated])
                
                original_data.append({
                    'doc': doc,
                    'metadata': metadata,
                    'distance': distance,
                    'index': i
                })
            
            # Get cross-encoder scores
            cross_encoder_scores = self.cross_encoder.predict(query_doc_pairs)
            
            # Combine with original data and sort by cross-encoder score
            scored_results = []
            for i, (score, data) in enumerate(zip(cross_encoder_scores, original_data)):
                scored_results.append({
                    'score': float(score),
                    'doc': data['doc'],
                    'metadata': data['metadata'],
                    'distance': data['distance'],
                    'original_index': data['index']
                })
            
            # Sort by cross-encoder score (highest first)
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Take top n_results
            top_results = scored_results[:n_results]
            
            # Reconstruct results format
            cross_encoder_results = {
                'ids': [results['ids'][0][item['original_index']] for item in top_results] if results.get('ids') else [[]],
                'documents': [[item['doc'] for item in top_results]],
                'metadatas': [[item['metadata'] for item in top_results]],
                'distances': [[item['distance'] for item in top_results]],  # Keep original distances
                'cross_encoder_scores': [[item['score'] for item in top_results]]  # Add cross-encoder scores
            }
            
            return cross_encoder_results
            
        except Exception as e:
            print(f"⚠️ Cross-encoder re-ranking failed: {e}")
            # Fallback to metadata re-ranking only
            return {
                'ids': [results['ids'][0][:n_results]] if results.get('ids') else [[]],
                'documents': [results['documents'][0][:n_results]],
                'metadatas': [results['metadatas'][0][:n_results]],
                'distances': [results['distances'][0][:n_results]]
            }
    
    def format_results(self, query_results: Dict[str, Any]) -> str:
        """Format query results for clinical display."""
        if 'error' in query_results:
            return f"❌ Error: {query_results['error']}"
        
        output = []
        output.append(f"🔍 Query: '{query_results['query']}'")
        if 'expanded_query' in query_results and query_results['expanded_query'] != query_results['query']:
            output.append(f"🔄 Expanded: '{query_results['expanded_query']}'")
        if 'using_medical_embeddings' in query_results and query_results['using_medical_embeddings']:
            output.append(f"🧠 Medical Embeddings: PubMedBERT-MS-MARCO (768d)")
        if 'using_cross_encoder' in query_results and query_results['using_cross_encoder']:
            output.append(f"🔄 Cross-Encoder Re-ranker: BioBERT-mnli (refined semantic matching)")
        if 'metadata_filters' in query_results and query_results['metadata_filters']:
            filter_info = ', '.join([f"{k}: {v}" for k, v in query_results['metadata_filters'].items()])
            output.append(f"🎯 Metadata Filters: {filter_info}")
        
        # Show explicit retrieval filters
        filters_applied = []
        if query_results.get('drug_filter'):
            filters_applied.append(f"Drug: {query_results['drug_filter']}")
        if query_results.get('section_filter'):
            filters_applied.append(f"Section: {query_results['section_filter']}")
        if filters_applied:
            output.append(f"🔍 Retrieval Filters: {', '.join(filters_applied)}")
        output.append("=" * 60)
        
        if not query_results['results']['documents'][0]:
            output.append("No relevant clinical data found.")
            return "\n".join(output)
        
        # Check if cross-encoder scores are available
        has_cross_encoder_scores = 'cross_encoder_scores' in query_results['results']
        cross_encoder_scores = query_results['results'].get('cross_encoder_scores', [[]])[0] if has_cross_encoder_scores else []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            query_results['results']['documents'][0],
            query_results['results']['metadatas'][0], 
            query_results['results']['distances'][0]
        )):
            relevance_score = 1 - distance
            if has_cross_encoder_scores and i < len(cross_encoder_scores):
                cross_score = cross_encoder_scores[i]
                output.append(f"\n📋 Result {i+1} (Semantic: {relevance_score:.3f}, Cross-Encoder: {cross_score:.3f})")
            else:
                output.append(f"\n📋 Result {i+1} (Relevance: {relevance_score:.3f})")
            
            # Enhanced metadata display
            output.append(f"📄 Document: {metadata['filename']}")
            output.append(f"📝 Type: {metadata['doc_type']}")
            output.append(f"🏛️ Source: {metadata['source']}")
            output.append(f"💊 Drugs: {metadata['drugs']}")
            output.append(f"🎯 Clinical Topic: {metadata.get('clinical_topic', 'General Clinical')}")
            
            # Enhanced metadata display
            if metadata.get('toxicity_grades'):
                output.append(f"⚠️ Toxicity Grades: {metadata['toxicity_grades']}")
            if metadata.get('laboratory_values'):
                output.append(f"🔬 Lab Values: {metadata['laboratory_values']}")
            if metadata.get('patient_population'):
                output.append(f"👥 Population: {metadata['patient_population']}")
            if metadata.get('treatment_phases'):
                output.append(f"⏱️ Treatment Phase: {metadata['treatment_phases']}")
            if metadata.get('evidence_level'):
                output.append(f"📊 Evidence: {metadata['evidence_level']}")
                
            output.append(f"📑 Chunk: {metadata['chunk_index']+1}/{metadata['total_chunks']}")
            output.append(f"🆔 Chunk ID: {metadata.get('chunk_id', 'N/A')}")
            
            # Extract section/page info from content if available
            section_info = self._extract_section_info(doc)
            if section_info:
                output.append(f"📍 Section: {section_info}")
            
            output.append(f"📖 Clinical Content:")
            
            # Show more complete content (500 characters)
            preview = doc[:500] + "..." if len(doc) > 500 else doc
            # Format content with proper indentation
            formatted_content = "\n".join([f"   {line}" for line in preview.split('\n')])
            output.append(formatted_content)
            output.append("-" * 60)
        
        return "\n".join(output)
    
    def _extract_section_info(self, content: str) -> str:
        """Extract section/heading information from content."""
        lines = content.split('\n')
        for line in lines[:5]:  # Check first 5 lines for headings
            line = line.strip()
            if line.startswith('##') and not line.startswith('###'):
                return line.replace('##', '').strip()
            elif line.startswith('###'):
                return line.replace('###', '').strip()
            elif line.startswith('#') and not line.startswith('##'):
                return line.replace('#', '').strip()
            elif line.startswith('**') and line.endswith('**') and len(line) < 100:
                return line.replace('**', '').strip()
        return ""
    
    def get_drug_specific_info(self, drug: str, topic: str = None) -> Dict[str, Any]:
        """Get drug-specific information with enhanced metadata filtering."""
        if topic:
            query = f"{drug} {topic}"
        else:
            query = drug
        
        # Get all results first, then filter by drug in post-processing
        # ChromaDB doesn't support $contains operator
        
        if self.embedding_model:
            query_embedding = self.embedding_model.encode([query])
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=20,  # Get more results to filter
                include=['documents', 'metadatas', 'distances']
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=20,  # Get more results to filter
                include=['documents', 'metadatas', 'distances']
            )
        
        # Filter results by drug in post-processing
        filtered_docs = []
        filtered_metadatas = []
        filtered_distances = []
        filtered_ids = []
        
        if results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                drugs_str = metadata.get('drugs', '').lower()
                if drug.lower() in drugs_str:
                    filtered_docs.append(doc)
                    filtered_metadatas.append(metadata)
                    filtered_distances.append(distance)
                    filtered_ids.append(results['ids'][0][i])
                    
                    if len(filtered_docs) >= 10:  # Limit to 10 results
                        break
        
        # Reconstruct results format
        filtered_results = {
            'ids': [filtered_ids],
            'documents': [filtered_docs],
            'metadatas': [filtered_metadatas],
            'distances': [filtered_distances]
        }
        
        return {
            'drug': drug,
            'topic': topic,
            'query': query,
            'results': filtered_results,
            'using_medical_embeddings': self.embedding_model is not None
        }
    
    def get_document_types(self) -> List[str]:
        """Get available document types."""
        sample = self.collection.get(limit=1000)
        doc_types = set()
        
        if sample['metadatas']:
            for metadata in sample['metadatas']:
                doc_types.add(metadata.get('doc_type', 'Unknown'))
        
        return sorted(list(doc_types))
    
    def interactive_query(self):
        """Interactive command-line interface for clinical queries."""
        print("\n🩺 GBM Clinical Query Interface")
        print("Type 'help' for commands, 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 Clinical Query: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thank you for using GBM Clinical Query Interface!")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                
                elif user_input.lower() == 'stats':
                    self.show_stats()
                
                # Parse filter commands
                elif user_input.startswith('filter:'):
                    self._handle_filter_command(user_input)
                
                elif user_input.startswith('tmz ') or user_input.startswith('temozolomide '):
                    topic = user_input.split(' ', 1)[1] if ' ' in user_input else None
                    results = self.get_drug_specific_info('temozolomide', topic)
                    print(self.format_results(results))
                
                elif user_input.startswith('avastin ') or user_input.startswith('bevacizumab '):
                    topic = user_input.split(' ', 1)[1] if ' ' in user_input else None
                    results = self.get_drug_specific_info('bevacizumab', topic)
                    print(self.format_results(results))
                
                elif user_input:
                    results = self.query_clinical_data(user_input)
                    print(self.format_results(results))
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _handle_filter_command(self, user_input: str):
        """Handle filter command parsing and execution."""
        try:
            # Parse filter command: filter:drug=temozolomide,section=dosing query text
            parts = user_input.split(' ', 1)
            if len(parts) < 2:
                print("❌ Usage: filter:drug=drugname,section=sectionname query text")
                return
            
            filter_part = parts[0].replace('filter:', '')
            query_text = parts[1]
            
            # Parse filters
            drug_filter = None
            section_filter = None
            
            for filter_item in filter_part.split(','):
                if '=' in filter_item:
                    key, value = filter_item.split('=', 1)
                    if key.strip() == 'drug':
                        drug_filter = value.strip()
                    elif key.strip() == 'section':
                        section_filter = value.strip()
            
            if not drug_filter and not section_filter:
                print("❌ No valid filters specified. Use drug=drugname or section=sectionname")
                return
            
            print(f"🔍 Applying filters - Drug: {drug_filter or 'None'}, Section: {section_filter or 'None'}")
            
            # Execute filtered query
            results = self.query_clinical_data(query_text, drug_filter=drug_filter, section_filter=section_filter)
            print(self.format_results(results))
            
        except Exception as e:
            print(f"❌ Filter command error: {e}")
            print("Usage: filter:drug=temozolomide,section=dose_modifications your query here")
    
    def show_help(self):
        """Show help information."""
        help_text = """
🩺 GBM Clinical Query Interface - Help

BASIC QUERIES:
- Type any clinical question about GBM treatment
- Example: "temozolomide dosing for newly diagnosed GBM"
- Example: "bevacizumab side effects monitoring"

DRUG-SPECIFIC QUERIES:
- tmz [topic] - Temozolomide-specific information
- temozolomide [topic] - Same as above
- avastin [topic] - Bevacizumab-specific information  
- bevacizumab [topic] - Same as above

COMMANDS:
- help - Show this help message
- stats - Show database statistics
- quit/exit/q - Exit the interface

EXAMPLES:
- "What is the standard dosing for newly diagnosed GBM?"
- "tmz maintenance protocol"
- "avastin contraindications"
- "FDA approval trials for bevacizumab"
- "side effects monitoring requirements"

FILTER COMMANDS:
- filter:drug=temozolomide [query] - Only TMZ-containing documents
- filter:drug=bevacizumab [query] - Only Avastin-containing documents  
- filter:section=dose_modifications [query] - Only dose modification content
- filter:section=contraindications [query] - Only contraindication content
- filter:drug=tmz,section=dosing [query] - Combined filters

FEATURES:
- 🧠 Medical domain embeddings (PubMedBERT)
- 🔄 Cross-encoder re-ranking for refined results
- 🎯 Metadata-based filtering and boosting
- 🔍 Explicit drug and section filtering
- 📊 Clinical terminology expansion
"""
        print(help_text)
    
    def show_stats(self):
        """Show database statistics."""
        total_count = self.collection.count()
        sample = self.collection.get(limit=100)
        
        doc_types = {}
        sources = {}
        drugs = {}
        
        if sample['metadatas']:
            for metadata in sample['metadatas']:
                # Count document types
                doc_type = metadata.get('doc_type', 'Unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                # Count sources
                source = metadata.get('source', 'Unknown')
                sources[source] = sources.get(source, 0) + 1
                
                # Count drugs
                drug_list = metadata.get('drugs', 'both').split(',')
                for drug in drug_list:
                    drugs[drug] = drugs.get(drug, 0) + 1
        
        print(f"\n📊 Database Statistics:")
        print(f"📋 Total chunks: {total_count}")
        print(f"📁 Document types: {doc_types}")
        print(f"🏛️ Sources: {sources}")
        print(f"💊 Drugs: {drugs}")

def main():
    """Main function to start the clinical query interface."""
    try:
        interface = ClinicalQueryInterface()
        interface.interactive_query()
    except Exception as e:
        print(f"❌ Failed to initialize interface: {e}")
        print("Make sure the vector database has been created by running create_vector_db.py first")

if __name__ == "__main__":
    main()
