#!/usr/bin/env python3
"""
Enhanced Metadata Filtering System for GBM Clinical Query Interface
Provides rich filtering by document type, clinical topic, evidence level, and more
Author: Chetanya Pandey
"""

from typing import Dict, List, Any, Optional, Set
import chromadb
from chromadb.config import Settings

class EnhancedMetadataFilter:
    def __init__(self, db_dir: str = "vector_db"):
        """Initialize enhanced metadata filtering system."""
        self.db_dir = db_dir
        
        # Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.chroma_client.get_collection("gbm_clinical_medical_embeddings")
        except:
            self.collection = self.chroma_client.get_collection("gbm_clinical_data")
        
        # Initialize available filter options by analyzing database
        self._init_filter_options()
        
        # Define filter hierarchies and relationships
        self._init_filter_hierarchies()
    
    def _init_filter_options(self):
        """Initialize available filter options by analyzing the database."""
        # Sample a representative set of documents to understand available metadata
        sample_size = min(1000, self.collection.count())
        sample = self.collection.get(limit=sample_size, include=['metadatas'])
        
        # Extract unique values for each metadata field
        self.available_filters = {
            'doc_types': set(),
            'sources': set(),
            'clinical_topics': set(),
            'evidence_levels': set(),
            'drugs': set(),
            'treatment_phases': set(),
            'patient_populations': set(),
            'toxicity_grades': set()
        }
        
        if sample['metadatas']:
            for metadata in sample['metadatas']:
                # Document types
                if metadata.get('doc_type'):
                    self.available_filters['doc_types'].add(metadata['doc_type'])
                
                # Sources
                if metadata.get('source'):
                    self.available_filters['sources'].add(metadata['source'])
                
                # Clinical topics
                if metadata.get('clinical_topic'):
                    self.available_filters['clinical_topics'].add(metadata['clinical_topic'])
                
                # Evidence levels
                if metadata.get('evidence_level'):
                    self.available_filters['evidence_levels'].add(metadata['evidence_level'])
                
                # Drugs
                if metadata.get('drugs'):
                    drugs = metadata['drugs'].split(',')
                    for drug in drugs:
                        self.available_filters['drugs'].add(drug.strip())
                
                # Treatment phases
                if metadata.get('treatment_phases'):
                    phases = metadata['treatment_phases'].split(',')
                    for phase in phases:
                        self.available_filters['treatment_phases'].add(phase.strip())
                
                # Patient populations
                if metadata.get('patient_population'):
                    self.available_filters['patient_populations'].add(metadata['patient_population'])
                
                # Toxicity grades
                if metadata.get('toxicity_grades'):
                    self.available_filters['toxicity_grades'].add(metadata['toxicity_grades'])
        
        # Convert sets to sorted lists for consistent display
        for key in self.available_filters:
            self.available_filters[key] = sorted(list(self.available_filters[key]))
    
    def _init_filter_hierarchies(self):
        """Initialize filter hierarchies and relationships."""
        # Document type hierarchy (most authoritative first)
        self.doc_type_hierarchy = [
            'prescribing_information',
            'fda_label', 
            'clinical_guideline',
            'clinical_protocol',
            'clinical_trial',
            'research_article',
            'hospital_protocol',
            'case_report'
        ]
        
        # Evidence level hierarchy
        self.evidence_hierarchy = [
            'fda_approved',
            'clinical_guideline',
            'randomized_controlled_trial',
            'clinical_trial',
            'systematic_review',
            'case_series',
            'expert_opinion'
        ]
        
        # Clinical topic groupings
        self.topic_groups = {
            'dosing': ['dosing', 'dose_calculation', 'administration', 'protocol'],
            'safety': ['toxicity', 'adverse_effects', 'monitoring', 'contraindications'],
            'efficacy': ['efficacy', 'survival', 'response', 'outcomes'],
            'administration': ['administration', 'preparation', 'infusion', 'timing'],
            'interactions': ['drug_interactions', 'concurrent_therapy', 'contraindications']
        }
        
        # Drug synonyms and variants
        self.drug_variants = {
            'temozolomide': ['temozolomide', 'tmz', 'temodar', 'temodal'],
            'bevacizumab': ['bevacizumab', 'avastin', 'anti-vegf']
        }
    
    def get_available_filters(self) -> Dict[str, List[str]]:
        """Get all available filter options."""
        return {
            'document_types': self.available_filters['doc_types'],
            'sources': self.available_filters['sources'],
            'clinical_topics': self.available_filters['clinical_topics'],
            'evidence_levels': self.available_filters['evidence_levels'],
            'drugs': self.available_filters['drugs'],
            'treatment_phases': self.available_filters['treatment_phases'],
            'patient_populations': self.available_filters['patient_populations'],
            'toxicity_grades': self.available_filters['toxicity_grades']
        }
    
    def build_metadata_query(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB metadata query from filter specifications."""
        where_conditions = {}
        
        # Document type filter
        if filters.get('doc_type'):
            doc_types = filters['doc_type'] if isinstance(filters['doc_type'], list) else [filters['doc_type']]
            if len(doc_types) == 1:
                where_conditions['doc_type'] = {'$eq': doc_types[0]}
            else:
                where_conditions['doc_type'] = {'$in': doc_types}
        
        # Source filter
        if filters.get('source'):
            sources = filters['source'] if isinstance(filters['source'], list) else [filters['source']]
            if len(sources) == 1:
                where_conditions['source'] = {'$eq': sources[0]}
            else:
                where_conditions['source'] = {'$in': sources}
        
        # Clinical topic filter
        if filters.get('clinical_topic'):
            topics = filters['clinical_topic'] if isinstance(filters['clinical_topic'], list) else [filters['clinical_topic']]
            if len(topics) == 1:
                where_conditions['clinical_topic'] = {'$eq': topics[0]}
            else:
                where_conditions['clinical_topic'] = {'$in': topics}
        
        # Evidence level filter
        if filters.get('evidence_level'):
            levels = filters['evidence_level'] if isinstance(filters['evidence_level'], list) else [filters['evidence_level']]
            if len(levels) == 1:
                where_conditions['evidence_level'] = {'$eq': levels[0]}
            else:
                where_conditions['evidence_level'] = {'$in': levels}
        
        # Drug filter (handle synonyms)
        if filters.get('drug'):
            drug_variants = self._expand_drug_variants(filters['drug'])
            # Use post-processing for drug filtering since it may contain multiple drugs
            pass
        
        # Treatment phase filter
        if filters.get('treatment_phase'):
            phases = filters['treatment_phase'] if isinstance(filters['treatment_phase'], list) else [filters['treatment_phase']]
            if len(phases) == 1:
                where_conditions['treatment_phases'] = {'$eq': phases[0]}
            else:
                where_conditions['treatment_phases'] = {'$in': phases}
        
        # Patient population filter
        if filters.get('patient_population'):
            pops = filters['patient_population'] if isinstance(filters['patient_population'], list) else [filters['patient_population']]
            if len(pops) == 1:
                where_conditions['patient_population'] = {'$eq': pops[0]}
            else:
                where_conditions['patient_population'] = {'$in': pops}
        
        return where_conditions if where_conditions else None
    
    def _expand_drug_variants(self, drug_input: str) -> List[str]:
        """Expand drug names to include variants and synonyms."""
        drug_lower = drug_input.lower()
        variants = []
        
        for main_drug, variant_list in self.drug_variants.items():
            if drug_lower in [v.lower() for v in variant_list]:
                variants.extend(variant_list)
                break
        
        return variants if variants else [drug_input]
    
    def apply_post_filters(self, results: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply post-processing filters that can't be handled by ChromaDB metadata queries."""
        if not results.get('documents', [[]])[0] or not filters:
            return results
        
        filtered_docs = []
        filtered_metadatas = []
        filtered_distances = []
        filtered_ids = []
        
        # Process drug filters (since drugs field may contain multiple comma-separated values)
        drug_filter = filters.get('drug')
        if drug_filter:
            drug_variants = self._expand_drug_variants(drug_filter)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            include_result = True
            
            # Apply drug filter
            if drug_filter and include_result:
                drugs_in_metadata = metadata.get('drugs', '').lower()
                drug_found = any(variant.lower() in drugs_in_metadata for variant in drug_variants)
                if not drug_found:
                    include_result = False
            
            # Apply toxicity grade filter
            if filters.get('toxicity_grade') and include_result:
                required_grade = filters['toxicity_grade'].lower()
                doc_grades = metadata.get('toxicity_grades', '').lower()
                if required_grade not in doc_grades and required_grade not in doc.lower():
                    include_result = False
            
            # Apply content-based filters
            if filters.get('contains_text') and include_result:
                required_text = filters['contains_text'].lower()
                if required_text not in doc.lower():
                    include_result = False
            
            if include_result:
                filtered_docs.append(doc)
                filtered_metadatas.append(metadata)
                filtered_distances.append(distance)
                filtered_ids.append(results.get('ids', [[]])[0][i] if results.get('ids') else f"filtered_{i}")
        
        return {
            'ids': [filtered_ids],
            'documents': [filtered_docs],
            'metadatas': [filtered_metadatas],
            'distances': [filtered_distances]
        }
    
    def suggest_filter_combinations(self, query: str) -> List[Dict[str, Any]]:
        """Suggest useful filter combinations based on the query."""
        query_lower = query.lower()
        suggestions = []
        
        # Drug-specific suggestions
        if any(drug in query_lower for drug in ['temozolomide', 'tmz', 'temodar']):
            suggestions.append({
                'name': 'TMZ FDA-Approved Information',
                'filters': {'drug': 'temozolomide', 'evidence_level': 'fda_approved'},
                'description': 'Official FDA-approved temozolomide prescribing information'
            })
            
            suggestions.append({
                'name': 'TMZ Clinical Guidelines',
                'filters': {'drug': 'temozolomide', 'doc_type': 'clinical_guideline'},
                'description': 'Clinical practice guidelines for temozolomide use'
            })
        
        if any(drug in query_lower for drug in ['bevacizumab', 'avastin']):
            suggestions.append({
                'name': 'Avastin FDA Information',
                'filters': {'drug': 'bevacizumab', 'evidence_level': 'fda_approved'},
                'description': 'FDA-approved bevacizumab prescribing information'
            })
        
        # Topic-specific suggestions
        if any(term in query_lower for term in ['dose', 'dosing', 'mg/m²', 'protocol']):
            suggestions.append({
                'name': 'Dosing Protocols Only',
                'filters': {'clinical_topic': 'dosing'},
                'description': 'Focus on dosing and administration protocols'
            })
        
        if any(term in query_lower for term in ['toxicity', 'side effects', 'adverse', 'monitoring']):
            suggestions.append({
                'name': 'Safety and Monitoring',
                'filters': {'clinical_topic': 'toxicity'},
                'description': 'Safety information and monitoring guidelines'
            })
        
        # Evidence level suggestions
        if 'fda' in query_lower or 'approved' in query_lower:
            suggestions.append({
                'name': 'FDA-Approved Information Only',
                'filters': {'evidence_level': 'fda_approved'},
                'description': 'Restrict to FDA-approved prescribing information'
            })
        
        # Population-specific suggestions
        if any(term in query_lower for term in ['elderly', 'old', 'geriatric']):
            suggestions.append({
                'name': 'Elderly Patient Information',
                'filters': {'patient_population': 'elderly'},
                'description': 'Information specific to elderly patients'
            })
        
        if any(term in query_lower for term in ['newly diagnosed', 'initial', 'first-line']):
            suggestions.append({
                'name': 'Newly Diagnosed GBM',
                'filters': {'treatment_phase': 'first_line'},
                'description': 'First-line treatment for newly diagnosed GBM'
            })
        
        # High-quality evidence suggestion
        suggestions.append({
            'name': 'High-Quality Evidence',
            'filters': {'evidence_level': ['fda_approved', 'clinical_guideline', 'randomized_controlled_trial']},
            'description': 'Restrict to highest quality evidence sources'
        })
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def format_filter_options(self) -> str:
        """Format available filter options for display."""
        output = []
        output.append("🔍 Available Metadata Filters")
        output.append("=" * 40)
        
        filters = self.get_available_filters()
        
        if filters['document_types']:
            output.append(f"\n📄 Document Types ({len(filters['document_types'])}):")
            for doc_type in filters['document_types'][:10]:  # Show first 10
                output.append(f"  • {doc_type}")
            if len(filters['document_types']) > 10:
                output.append(f"  ... and {len(filters['document_types']) - 10} more")
        
        if filters['clinical_topics']:
            output.append(f"\n🎯 Clinical Topics ({len(filters['clinical_topics'])}):")
            for topic in filters['clinical_topics'][:8]:
                output.append(f"  • {topic}")
            if len(filters['clinical_topics']) > 8:
                output.append(f"  ... and {len(filters['clinical_topics']) - 8} more")
        
        if filters['evidence_levels']:
            output.append(f"\n📊 Evidence Levels ({len(filters['evidence_levels'])}):")
            for level in filters['evidence_levels'][:6]:
                output.append(f"  • {level}")
        
        if filters['drugs']:
            output.append(f"\n💊 Available Drugs ({len(filters['drugs'])}):")
            for drug in filters['drugs']:
                output.append(f"  • {drug}")
        
        if filters['sources']:
            output.append(f"\n🏛️ Sources ({len(filters['sources'])}):")
            for source in filters['sources'][:8]:
                output.append(f"  • {source}")
            if len(filters['sources']) > 8:
                output.append(f"  ... and {len(filters['sources']) - 8} more")
        
        output.append(f"\n💡 Usage Examples:")
        output.append(f"  filter:doc_type=prescribing_information your query")
        output.append(f"  filter:drug=temozolomide,topic=dosing your query")
        output.append(f"  filter:evidence=fda_approved your query")
        
        return "\n".join(output)
    
    def get_filter_statistics(self) -> Dict[str, int]:
        """Get statistics about filter usage and document distribution."""
        stats = {}
        
        # Get total document count
        total_docs = self.collection.count()
        stats['total_documents'] = total_docs
        
        # Get distribution by major categories
        filters = self.get_available_filters()
        
        stats['document_types'] = len(filters['document_types'])
        stats['clinical_topics'] = len(filters['clinical_topics'])
        stats['evidence_levels'] = len(filters['evidence_levels'])
        stats['unique_sources'] = len(filters['sources'])
        stats['drugs_covered'] = len(filters['drugs'])
        
        return stats

def test_metadata_filter():
    """Test the enhanced metadata filtering system."""
    filter_system = EnhancedMetadataFilter()
    
    print("🧪 Testing Enhanced Metadata Filter System")
    print("=" * 50)
    
    # Test available filters
    print("\n1. Available Filters:")
    print(filter_system.format_filter_options())
    
    # Test filter suggestions
    test_queries = [
        "temozolomide dosing for newly diagnosed GBM",
        "bevacizumab side effects monitoring",
        "FDA approved protocols"
    ]
    
    print("\n2. Filter Suggestions:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        suggestions = filter_system.suggest_filter_combinations(query)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion['name']}")
            print(f"     Filters: {suggestion['filters']}")
            print(f"     Description: {suggestion['description']}")
    
    # Test metadata query building
    print("\n3. Metadata Query Building:")
    test_filters = {
        'drug': 'temozolomide',
        'clinical_topic': 'dosing',
        'evidence_level': 'fda_approved'
    }
    
    metadata_query = filter_system.build_metadata_query(test_filters)
    print(f"Filters: {test_filters}")
    print(f"Generated ChromaDB query: {metadata_query}")
    
    # Test statistics
    print("\n4. Filter Statistics:")
    stats = filter_system.get_filter_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_metadata_filter()