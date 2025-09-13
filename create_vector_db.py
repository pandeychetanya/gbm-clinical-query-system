#!/usr/bin/env python3
"""
GBM Clinical Data Vector Database Creator
Creates a ChromaDB vector database from clinical documents for retrieval.
Author: Chetanya Pandey
"""

import os
import glob
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from typing import List, Dict, Any
from datetime import datetime

class GBMVectorDB:
    def __init__(self, data_dir: str = "us_clinical_data", db_dir: str = "vector_db"):
        """
        Initialize the GBM Vector Database.
        
        Args:
            data_dir: Directory containing clinical markdown files
            db_dir: Directory to store the ChromaDB database
        """
        self.data_dir = data_dir
        self.db_dir = db_dir
        
        # Initialize ChromaDB
        os.makedirs(db_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create collection for GBM clinical data with medical embeddings
        collection_name = "gbm_clinical_medical_embeddings"
        try:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Temozolomide and Bevacizumab clinical data with medical domain embeddings",
                    "embedding_model": "medical_domain",
                    "created_at": datetime.now().isoformat()
                }
            )
        except Exception:
            # Collection already exists - try to get it
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
            except:
                # Delete and recreate if there's a conflict
                try:
                    self.chroma_client.delete_collection(collection_name)
                    self.collection = self.chroma_client.create_collection(
                        name=collection_name,
                        metadata={
                            "description": "Temozolomide and Bevacizumab clinical data with medical domain embeddings",
                            "embedding_model": "medical_domain",
                            "created_at": datetime.now().isoformat()
                        }
                    )
                except Exception as e:
                    print(f"Warning: Collection creation issue: {e}")
                    self.collection = self.chroma_client.get_or_create_collection(collection_name)
        
        # Initialize medical domain embedding model
        print("Loading medical domain SentenceTransformer model...")
        
        # Try medical domain models in order of preference
        medical_models = [
            'pritamdeka/S-PubMedBert-MS-MARCO',  # PubMedBERT fine-tuned for retrieval
            'pritamdeka/S-BioBert-snli-multinli-stsb',  # BioBERT fine-tuned for sentence similarity
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',  # PubMedBERT base
            'dmis-lab/biobert-v1.1',  # BioBERT
            'emilyalsentzer/Bio_ClinicalBERT',  # ClinicalBERT
            'all-MiniLM-L6-v2'  # Fallback general model
        ]
        
        self.embedding_model = None
        for model_name in medical_models:
            try:
                print(f"Attempting to load: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
                print(f"✅ Successfully loaded medical model: {model_name}")
                break
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        if self.embedding_model is None:
            raise Exception("Failed to load any embedding model")
        
        # Initialize text splitter for clinical content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # Smaller chunks for focused clinical concepts
            chunk_overlap=100,  # Reduced overlap
            separators=[
                "\n## ",      # Major sections (dosing, toxicity, etc.)
                "\n### ",     # Subsections (specific protocols)
                "\n#### ",    # Clinical details
                "\n**",       # Bold clinical concepts
                "\n- **",     # Bullet points with bold headers
                "\n---",      # Section dividers
                "\n\n",       # Paragraph breaks
                "\n",         # Line breaks
                ". ",         # Sentences
                "! ",         # Exclamations
                "? ",         # Questions
                "; ",         # Semicolons
                ", ",         # Commas
                " ",          # Spaces
                ""            # Characters
            ]
        )
        
        print(f"✅ GBM Vector DB initialized")
        print(f"📁 Data directory: {data_dir}")
        print(f"🗄️ Database directory: {db_dir}")
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load all markdown documents from the data directory."""
        documents = []
        
        # Get all markdown files
        md_files = glob.glob(os.path.join(self.data_dir, "*.md"))
        
        print(f"📄 Found {len(md_files)} markdown files")
        
        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract metadata from filename
                filename = os.path.basename(file_path)
                doc_type = self._classify_document(filename)
                
                documents.append({
                    'filename': filename,
                    'filepath': file_path,
                    'content': content,
                    'doc_type': doc_type,
                    'drug': self._extract_drug_info(filename, content),
                    'source': self._extract_source_info(filename)
                })
                
                print(f"📋 Loaded: {filename} ({doc_type})")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        return documents
    
    def _classify_document(self, filename: str) -> str:
        """Classify document type based on filename."""
        filename_lower = filename.lower()
        
        if 'fda' in filename_lower and 'complete' in filename_lower:
            return 'FDA_Complete_Prescribing_Information'
        elif 'fda' in filename_lower and 'approval' in filename_lower:
            return 'FDA_Approval_Trials'
        elif 'fda' in filename_lower:
            return 'FDA_Document'
        elif 'nccn' in filename_lower:
            return 'NCCN_Guidelines'
        elif 'stupp' in filename_lower:
            return 'Pivotal_Clinical_Trial'
        elif 'hospital' in filename_lower or 'protocols' in filename_lower:
            return 'Hospital_Protocol'
        elif 'dailymed' in filename_lower:
            return 'DailyMed_Prescribing_Info'
        elif 'trial' in filename_lower or 'clinical' in filename_lower:
            return 'Clinical_Research'
        elif 'dosing' in filename_lower:
            return 'Dosing_Protocol'
        elif 'source' in filename_lower:
            return 'Source_Reference'
        else:
            return 'Clinical_Document'
    
    def _extract_drug_info(self, filename: str, content: str) -> List[str]:
        """Extract drug information from filename and content."""
        drugs = []
        
        # Check filename
        filename_lower = filename.lower()
        if 'temodar' in filename_lower or 'temozolomide' in filename_lower:
            drugs.append('temozolomide')
        if 'avastin' in filename_lower or 'bevacizumab' in filename_lower:
            drugs.append('bevacizumab')
        
        # Check content
        content_lower = content.lower()
        if 'temozolomide' in content_lower or 'temodar' in content_lower:
            if 'temozolomide' not in drugs:
                drugs.append('temozolomide')
        if 'bevacizumab' in content_lower or 'avastin' in content_lower:
            if 'bevacizumab' not in drugs:
                drugs.append('bevacizumab')
        
        return drugs if drugs else ['both']
    
    def _extract_source_info(self, filename: str) -> str:
        """Extract source information from filename."""
        filename_lower = filename.lower()
        
        if 'fda' in filename_lower:
            return 'FDA'
        elif 'nccn' in filename_lower:
            return 'NCCN'
        elif 'nejm' in filename_lower:
            return 'NEJM'
        elif 'dailymed' in filename_lower:
            return 'DailyMed'
        elif 'hospital' in filename_lower:
            return 'Hospital'
        else:
            return 'Clinical_Literature'
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into focused clinical chunks."""
        chunks = []
        
        for doc in documents:
            # First try clinical-focused chunking
            clinical_chunks = self._create_clinical_chunks(doc)
            
            if clinical_chunks:
                chunks.extend(clinical_chunks)
            else:
                # Fallback to standard chunking
                text_chunks = self.text_splitter.split_text(doc['content'])
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        'chunk_id': f"{doc['filename']}_chunk_{i}",
                        'content': chunk,
                        'filename': doc['filename'],
                        'doc_type': doc['doc_type'],
                        'drug': doc['drug'],
                        'source': doc['source'],
                        'chunk_index': i,
                        'total_chunks': len(text_chunks),
                        'clinical_topic': self._extract_clinical_topic(chunk)
                    })
        
        print(f"📄 Created {len(chunks)} focused clinical chunks from {len(documents)} documents")
        return chunks
    
    def _create_clinical_chunks(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks focused on clinical concepts."""
        content = doc['content']
        chunks = []
        
        # Define clinical section patterns
        clinical_patterns = [
            # Dosing sections
            (r'#{1,4}\s*.*[Dd]os[ei]ng.*', 'Dosing Protocol'),
            (r'#{1,4}\s*.*[Aa]dministration.*', 'Administration'),
            (r'#{1,4}\s*.*[Mm]aintenance.*', 'Maintenance Protocol'),
            (r'#{1,4}\s*.*[Cc]oncomitant.*', 'Concomitant Protocol'),
            
            # Safety sections
            (r'#{1,4}\s*.*[Tt]oxicity.*', 'Toxicity Profile'),
            (r'#{1,4}\s*.*[Ss]ide [Ee]ffects.*', 'Side Effects'),
            (r'#{1,4}\s*.*[Aa]dverse.*', 'Adverse Events'),
            (r'#{1,4}\s*.*[Mm]ortality.*', 'Mortality Data'),
            (r'#{1,4}\s*.*[Ss]afety.*', 'Safety Monitoring'),
            (r'#{1,4}\s*.*[Mm]onitoring.*', 'Clinical Monitoring'),
            
            # Modification sections
            (r'#{1,4}\s*.*[Dd]ose [Mm]odification.*', 'Dose Modifications'),
            (r'#{1,4}\s*.*[Rr]eduction.*', 'Dose Reduction'),
            
            # Specific conditions
            (r'#{1,4}\s*.*[Hh]ematologic.*', 'Hematologic Effects'),
            (r'#{1,4}\s*.*[Cc]ardiovascular.*', 'Cardiovascular Effects'),
            (r'#{1,4}\s*.*[Nn]eurologic.*', 'Neurologic Effects'),
            (r'#{1,4}\s*.*[Gg]astrointestinal.*', 'GI Effects'),
            
            # Clinical management
            (r'#{1,4}\s*.*[Pp]rotocol.*', 'Clinical Protocol'),
            (r'#{1,4}\s*.*[Gg]uidelines.*', 'Clinical Guidelines'),
            (r'#{1,4}\s*.*[Mm]anagement.*', 'Clinical Management'),
            
            # Drug-specific sections
            (r'#{1,4}\s*.*[Tt]emozolomide.*', 'Temozolomide Info'),
            (r'#{1,4}\s*.*[Bb]evacizumab.*', 'Bevacizumab Info'),
            (r'#{1,4}\s*.*[Aa]vastin.*', 'Avastin Info'),
        ]
        
        # Use improved text splitter
        text_chunks = self.text_splitter.split_text(content)
        
        for i, chunk in enumerate(text_chunks):
            # Determine clinical topic
            clinical_topic = self._extract_clinical_topic(chunk)
            
            # Extract detailed metadata
            detailed_metadata = self._extract_detailed_metadata(chunk)
            
            chunk_data = {
                'chunk_id': f"{doc['filename']}_clinical_{i}",
                'content': chunk,
                'filename': doc['filename'],
                'doc_type': doc['doc_type'],
                'drug': doc['drug'],
                'source': doc['source'],
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'clinical_topic': clinical_topic
            }
            
            # Add detailed metadata
            chunk_data.update(detailed_metadata)
            chunks.append(chunk_data)
        
        return chunks
    
    def _extract_clinical_topic(self, chunk: str) -> str:
        """Extract the main clinical topic from a chunk."""
        chunk_lower = chunk.lower()
        
        # Dosing-related topics
        if any(word in chunk_lower for word in ['dosing', 'dose', 'mg/m²', 'mg/kg']):
            if 'maintenance' in chunk_lower:
                return 'Maintenance Dosing'
            elif 'concomitant' in chunk_lower or 'concurrent' in chunk_lower:
                return 'Concomitant Dosing'
            elif 'modification' in chunk_lower or 'reduction' in chunk_lower:
                return 'Dose Modifications'
            else:
                return 'Dosing Protocol'
        
        # Toxicity-related topics
        elif any(word in chunk_lower for word in ['toxicity', 'adverse', 'side effect', 'mortality', 'death']):
            if 'hematologic' in chunk_lower or 'neutropenia' in chunk_lower or 'thrombocytopenia' in chunk_lower:
                return 'Hematologic Toxicity'
            elif 'cardiovascular' in chunk_lower or 'hypertension' in chunk_lower:
                return 'Cardiovascular Toxicity'
            elif 'mortality' in chunk_lower or 'death' in chunk_lower:
                return 'Treatment Mortality'
            else:
                return 'Toxicity Profile'
        
        # Monitoring topics
        elif any(word in chunk_lower for word in ['monitoring', 'laboratory', 'cbc', 'blood pressure']):
            return 'Clinical Monitoring'
        
        # Administration topics
        elif any(word in chunk_lower for word in ['administration', 'infusion', 'oral', 'iv']):
            return 'Drug Administration'
        
        # Clinical management
        elif any(word in chunk_lower for word in ['protocol', 'guideline', 'management']):
            return 'Clinical Protocol'
        
        # Drug-specific
        elif 'temozolomide' in chunk_lower or 'temodar' in chunk_lower:
            return 'Temozolomide Specific'
        elif 'bevacizumab' in chunk_lower or 'avastin' in chunk_lower:
            return 'Bevacizumab Specific'
        
        # Default
        else:
            return 'General Clinical'
    
    def _extract_detailed_metadata(self, chunk: str) -> Dict[str, Any]:
        """Extract detailed structured metadata from clinical chunk."""
        chunk_lower = chunk.lower()
        metadata = {}
        
        # Extract section information
        lines = chunk.split('\n')
        section = None
        subsection = None
        
        for line in lines[:5]:  # Check first 5 lines
            line_clean = line.strip()
            if line_clean.startswith('### '):
                subsection = line_clean.replace('###', '').strip()
            elif line_clean.startswith('## '):
                section = line_clean.replace('##', '').strip()
            elif line_clean.startswith('# ') and not line_clean.startswith('##'):
                section = line_clean.replace('#', '').strip()
            elif line_clean.startswith('**') and line_clean.endswith('**') and len(line_clean) < 80:
                if not subsection:
                    subsection = line_clean.replace('**', '').strip()
        
        metadata['section'] = section or 'Unknown'
        metadata['subsection'] = subsection or 'General'
        
        # Extract dosing information
        dosing_patterns = {
            'mg_per_m2': r'(\d+(?:\.\d+)?)\s*mg/m[²2]',
            'mg_per_kg': r'(\d+(?:\.\d+)?)\s*mg/kg',
            'cycle_length': r'(\d+)[-\s]*day',
            'frequency': r'every\s*(\d+)\s*weeks?|q(\d+)w'
        }
        
        for key, pattern in dosing_patterns.items():
            import re
            matches = re.findall(pattern, chunk_lower)
            if matches:
                if key == 'frequency':
                    # Handle both "every X weeks" and "qXw" patterns
                    metadata[key] = [m[0] or m[1] for m in matches if any(m)]
                else:
                    metadata[key] = matches
        
        # Extract clinical grade information
        grade_matches = re.findall(r'grade\s*(\d+)', chunk_lower)
        if grade_matches:
            metadata['toxicity_grades'] = list(set(grade_matches))
        
        # Extract laboratory values
        lab_patterns = {
            'anc_values': r'anc[:\s]*(?:≥|>=|>|<|≤|<=)\s*([0-9.,]+)',
            'platelet_values': r'platelet[s]?[:\s]*(?:≥|>=|>|<|≤|<=)\s*([0-9.,]+)',
            'hemoglobin_values': r'h[gb|emoglobin][:\s]*(?:≥|>=|>|<|≤|<=)\s*([0-9.,]+)'
        }
        
        for key, pattern in lab_patterns.items():
            matches = re.findall(pattern, chunk_lower)
            if matches:
                metadata[key] = matches
        
        # Extract patient population
        population_indicators = {
            'newly_diagnosed': ['newly diagnosed', 'initial', 'upfront', 'first-line'],
            'recurrent': ['recurrent', 'progressive', 'relapsed', 'refractory', 'salvage'],
            'elderly': ['elderly', '≥70', '>70', '≥65', '>65', 'geriatric'],
            'poor_performance': ['kps', 'performance status', 'ecog']
        }
        
        for pop_type, indicators in population_indicators.items():
            if any(indicator in chunk_lower for indicator in indicators):
                metadata[f'population_{pop_type}'] = True
        
        # Extract treatment phase
        treatment_phases = {
            'concomitant': ['concomitant', 'concurrent', 'chemoradiation', 'during radiation'],
            'maintenance': ['maintenance', 'adjuvant', 'post-radiation'],
            'salvage': ['salvage', 'rescue', 'recurrent treatment']
        }
        
        for phase, indicators in treatment_phases.items():
            if any(indicator in chunk_lower for indicator in indicators):
                metadata[f'treatment_phase_{phase}'] = True
        
        # Extract evidence level
        evidence_indicators = {
            'fda_approved': ['fda approved', 'fda approval', 'approved by fda'],
            'clinical_trial': ['phase ii', 'phase iii', 'clinical trial', 'randomized'],
            'guideline': ['nccn', 'guideline', 'recommendation', 'standard of care'],
            'real_world': ['real world', 'clinical practice', 'institutional']
        }
        
        for evidence_type, indicators in evidence_indicators.items():
            if any(indicator in chunk_lower for indicator in indicators):
                metadata[f'evidence_{evidence_type}'] = True
        
        return metadata
    
    def create_embeddings_and_store(self, chunks: List[Dict[str, Any]]) -> None:
        """Create medical domain embeddings and store in ChromaDB."""
        print("🔄 Creating medical domain embeddings and storing in vector database...")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        print(f"📊 Processing {len(chunks)} chunks with medical embeddings...")
        
        # Process chunks in batches to avoid memory issues
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_texts = [chunk['content'] for chunk in batch_chunks]
            
            # Create medical domain embeddings
            print(f"🧠 Creating embeddings for batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Add to collections
            for j, chunk in enumerate(batch_chunks):
                documents.append(chunk['content'])
                ids.append(chunk['chunk_id'])
                embeddings.append(batch_embeddings[j].tolist())
                
                # Base metadata
                metadata = {
                    'filename': chunk['filename'],
                    'doc_type': chunk['doc_type'],
                    'source': chunk['source'],
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'drugs': ','.join(chunk['drug']),
                    'clinical_topic': chunk.get('clinical_topic', 'General Clinical'),
                    'embedding_model': self.embedding_model.get_sentence_embedding_dimension(),
                    'created_at': datetime.now().isoformat()
                }
                
                # Add detailed clinical metadata
                detailed_fields = [
                    'section', 'subsection', 'mg_per_m2', 'mg_per_kg', 'cycle_length', 
                    'frequency', 'toxicity_grades', 'anc_values', 'platelet_values', 
                    'hemoglobin_values', 'population_newly_diagnosed', 'population_recurrent',
                    'population_elderly', 'population_poor_performance', 'treatment_phase_concomitant',
                    'treatment_phase_maintenance', 'treatment_phase_salvage', 'evidence_fda_approved',
                    'evidence_clinical_trial', 'evidence_guideline', 'evidence_real_world'
                ]
                
                for field in detailed_fields:
                    if field in chunk:
                        # Convert lists to strings for ChromaDB storage
                        value = chunk[field]
                        if isinstance(value, list):
                            metadata[field] = ','.join(map(str, value))
                        elif isinstance(value, bool):
                            metadata[field] = str(value)
                        else:
                            metadata[field] = str(value)
                    else:
                        metadata[field] = ''
                metadatas.append(metadata)
        
        # Add to collection with custom medical embeddings
        print("💾 Storing medical embeddings in ChromaDB...")
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        print(f"✅ Stored {len(chunks)} chunks with medical domain embeddings")
        print(f"🧠 Embedding model: {type(self.embedding_model).__name__}")
        print(f"📏 Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        print(f"🗄️ Database location: {self.db_dir}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample = self.collection.get(limit=100)
            
            # Analyze document types
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
            
            return {
                'total_chunks': count,
                'doc_types': doc_types,
                'sources': sources,
                'drugs': drugs,
                'collection_name': self.collection.name
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def test_search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Test search functionality."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            return {
                'query': query,
                'n_results': n_results,
                'results': results
            }
        
        except Exception as e:
            return {'error': str(e)}

def main():
    """Main function to create the vector database."""
    print("🚀 Starting GBM Clinical Data Vector Database Creation")
    print("=" * 60)
    
    # Initialize vector DB
    vector_db = GBMVectorDB()
    
    # Load documents
    print("\\n📄 Loading clinical documents...")
    documents = vector_db.load_documents()
    
    if not documents:
        print("❌ No documents found! Make sure markdown files exist in us_clinical_data/")
        return
    
    # Chunk documents
    print("\\n✂️ Chunking documents...")
    chunks = vector_db.chunk_documents(documents)
    
    # Create embeddings and store
    print("\\n🔄 Creating embeddings and storing in database...")
    vector_db.create_embeddings_and_store(chunks)
    
    # Get database statistics
    print("\\n📊 Database Statistics:")
    stats = vector_db.get_database_stats()
    
    if 'error' not in stats:
        print(f"📋 Total chunks stored: {stats['total_chunks']}")
        print(f"📁 Document types: {stats['doc_types']}")
        print(f"🏛️ Sources: {stats['sources']}")
        print(f"💊 Drugs covered: {stats['drugs']}")
    else:
        print(f"❌ Error getting stats: {stats['error']}")
    
    # Test search functionality
    print("\\n🔍 Testing search functionality...")
    test_queries = [
        "temozolomide dosing for glioblastoma",
        "bevacizumab recurrent GBM dose",
        "FDA approval trial results",
        "maintenance treatment protocol",
        "side effects monitoring"
    ]
    
    for query in test_queries:
        print(f"\\n🔎 Query: '{query}'")
        results = vector_db.test_search(query, n_results=3)
        
        if 'error' not in results:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['results']['documents'][0],
                results['results']['metadatas'][0], 
                results['results']['distances'][0]
            )):
                print(f"  {i+1}. {metadata['filename']} ({metadata['doc_type']}) - Distance: {distance:.3f}")
                print(f"     Preview: {doc[:100]}...")
        else:
            print(f"     ❌ Search error: {results['error']}")
    
    print("\\n" + "=" * 60)
    print("✅ Vector database creation completed successfully!")
    print(f"🗄️ Database stored in: {vector_db.db_dir}")
    print("🔍 Ready for clinical queries!")

if __name__ == "__main__":
    main()