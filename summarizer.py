#!/usr/bin/env python3
"""
Clinical Summarization System for GBM Query Interface
Provides concise clinical summaries from top-ranked chunks using local and API-based models
Author: Chetanya Pandey
"""

from typing import List, Dict, Any, Optional
import json
import re
import os
from sentence_transformers import SentenceTransformer
import requests
import time

class ClinicalSummarizer:
    def __init__(self, use_local_model: bool = True, api_key: Optional[str] = None):
        """Initialize clinical summarization system."""
        self.use_local_model = use_local_model
        self.api_key = api_key
        
        # Local summarization model (lighter weight)
        if use_local_model:
            try:
                # Try to load a medical domain summarization model
                from transformers import pipeline
                self.local_summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",  # Good general summarizer
                    device=-1  # Use CPU
                )
                print("✅ Loaded local summarization model: BART-large-CNN")
            except Exception as e:
                print(f"⚠️ Failed to load local summarizer: {e}")
                self.local_summarizer = None
        else:
            self.local_summarizer = None
        
        # Clinical summarization templates
        self.clinical_templates = {
            'dosing': {
                'prompt': "Summarize the key dosing information for this clinical query:",
                'focus': ['dose', 'mg/m²', 'mg/kg', 'protocol', 'regimen', 'schedule', 'administration']
            },
            'toxicity': {
                'prompt': "Summarize the key toxicity and monitoring information:",
                'focus': ['adverse effects', 'side effects', 'toxicity', 'monitoring', 'CBC', 'laboratory']
            },
            'contraindications': {
                'prompt': "Summarize the contraindications and precautions:",
                'focus': ['contraindication', 'avoid', 'warning', 'precaution', 'not recommended']
            },
            'administration': {
                'prompt': "Summarize the administration guidelines:",
                'focus': ['administration', 'infusion', 'preparation', 'timing', 'premedication']
            },
            'general': {
                'prompt': "Provide a concise clinical summary of this information:",
                'focus': []
            }
        }
        
        # Key clinical information extractors
        self.clinical_extractors = {
            'dosing_values': re.compile(r'(\d+(?:\.\d+)?)\s*(mg/m²|mg/kg|mg)\b', re.IGNORECASE),
            'toxicity_grades': re.compile(r'\b(grade\s+[1-4]|ctc\s+grade\s+[1-4])\b', re.IGNORECASE),
            'lab_values': re.compile(r'(\d+,?\d*)\s*(platelets?|neutrophils?|hemoglobin|anc)\b', re.IGNORECASE),
            'frequencies': re.compile(r'\b(daily|weekly|monthly|every\s+\d+\s+days?|q\d+[hdw])\b', re.IGNORECASE),
            'clinical_actions': re.compile(r'\b(hold|withhold|discontinue|reduce|modify|interrupt)\b', re.IGNORECASE)
        }
    
    def summarize_clinical_results(self, query: str, results: Dict[str, Any], 
                                  max_chunks: int = 5) -> Dict[str, Any]:
        """Generate clinical summary from search results."""
        if not results.get('results', {}).get('documents', [[]])[0]:
            return {'summary': 'No relevant clinical information found.', 'confidence': 0}
        
        # Get top chunks
        documents = results['results']['documents'][0][:max_chunks]
        metadatas = results['results']['metadatas'][0][:max_chunks] if results['results'].get('metadatas') else []
        
        # Determine clinical category
        clinical_category = self._determine_clinical_category(query)
        
        # Extract and combine relevant information
        combined_content = self._extract_relevant_content(documents, metadatas, query, clinical_category)
        
        # Generate summary
        summary_result = self._generate_summary(combined_content, query, clinical_category)
        
        # Extract key clinical facts
        clinical_facts = self._extract_clinical_facts(combined_content, query)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(documents, metadatas, query)
        
        return {
            'summary': summary_result['summary'],
            'clinical_category': clinical_category,
            'key_facts': clinical_facts,
            'confidence_score': confidence,
            'source_count': len(documents),
            'evidence_sources': self._get_evidence_sources(metadatas),
            'generation_method': summary_result['method'],
            'warnings': self._identify_clinical_warnings(combined_content)
        }
    
    def _determine_clinical_category(self, query: str) -> str:
        """Determine the clinical category of the query."""
        query_lower = query.lower()
        
        # Check for specific clinical categories
        if any(term in query_lower for term in ['dose', 'dosing', 'mg/m²', 'protocol', 'regimen']):
            return 'dosing'
        elif any(term in query_lower for term in ['toxicity', 'side effects', 'adverse', 'monitoring', 'CBC']):
            return 'toxicity'
        elif any(term in query_lower for term in ['contraindication', 'avoid', 'warning', 'precaution']):
            return 'contraindications'
        elif any(term in query_lower for term in ['administration', 'infusion', 'give', 'prepare']):
            return 'administration'
        else:
            return 'general'
    
    def _extract_relevant_content(self, documents: List[str], metadatas: List[Dict], 
                                query: str, category: str) -> str:
        """Extract and combine relevant content from documents."""
        relevant_parts = []
        template = self.clinical_templates[category]
        focus_terms = template['focus']
        
        for doc, metadata in zip(documents, metadatas if metadatas else [{}] * len(documents)):
            # Score sentences by relevance to query and category
            sentences = re.split(r'[.!?]+', doc)
            scored_sentences = []
            
            for sentence in sentences:
                if len(sentence.strip()) < 20:  # Skip very short sentences
                    continue
                
                score = 0
                sentence_lower = sentence.lower()
                
                # Score by query terms
                query_words = query.lower().split()
                for word in query_words:
                    if word in sentence_lower:
                        score += 2
                
                # Score by category focus terms
                for term in focus_terms:
                    if term in sentence_lower:
                        score += 3
                
                # Score by clinical importance
                if any(pattern.search(sentence) for pattern in self.clinical_extractors.values()):
                    score += 1
                
                if score > 0:
                    scored_sentences.append((sentence.strip(), score))
            
            # Take top sentences from this document
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in scored_sentences[:3]]  # Top 3 sentences per doc
            
            if top_sentences:
                doc_summary = ' '.join(top_sentences)
                source_info = f"[{metadata.get('source', 'Unknown')}]"
                relevant_parts.append(f"{source_info} {doc_summary}")
        
        return ' '.join(relevant_parts)
    
    def _generate_summary(self, content: str, query: str, category: str) -> Dict[str, str]:
        """Generate clinical summary using available methods."""
        if not content.strip():
            return {'summary': 'Insufficient clinical information available.', 'method': 'none'}
        
        template = self.clinical_templates[category]
        
        # Try local model first if available
        if self.local_summarizer and len(content) > 100:
            try:
                # Prepare content for summarization
                max_length = min(1024, len(content))  # BART max input length
                truncated_content = content[:max_length]
                
                # Generate summary
                summary_result = self.local_summarizer(
                    truncated_content,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
                
                summary_text = summary_result[0]['summary_text']
                
                # Post-process for clinical relevance
                clinical_summary = self._post_process_summary(summary_text, query, category)
                
                return {'summary': clinical_summary, 'method': 'local_bart'}
                
            except Exception as e:
                print(f"⚠️ Local summarization failed: {e}")
        
        # Fallback to extractive summarization
        extractive_summary = self._extractive_summarization(content, query, category)
        return {'summary': extractive_summary, 'method': 'extractive'}
    
    def _extractive_summarization(self, content: str, query: str, category: str) -> str:
        """Create extractive summary by selecting key sentences."""
        sentences = re.split(r'[.!?]+', content)
        if not sentences:
            return "No specific clinical information available."
        
        # Score sentences for clinical relevance
        scored_sentences = []
        query_words = set(query.lower().split())
        template = self.clinical_templates[category]
        focus_terms = set(template['focus'])
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
                
            sentence_lower = sentence.lower()
            sentence_words = set(sentence_lower.split())
            
            score = 0
            
            # Query overlap score
            query_overlap = len(query_words.intersection(sentence_words))
            score += query_overlap * 2
            
            # Clinical category score
            category_overlap = len(focus_terms.intersection(sentence_words))
            score += category_overlap * 3
            
            # Clinical entity bonus
            for pattern in self.clinical_extractors.values():
                if pattern.search(sentence):
                    score += 2
            
            # Length penalty for very long sentences
            if len(sentence) > 200:
                score -= 1
            
            if score > 0:
                scored_sentences.append((sentence.strip(), score))
        
        # Select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 3-5 sentences depending on content
        num_sentences = min(5, max(3, len(scored_sentences) // 3))
        top_sentences = [s[0] for s in scored_sentences[:num_sentences]]
        
        if not top_sentences:
            return "Clinical information found but specific details for this query are limited."
        
        # Combine and clean up
        summary = ' '.join(top_sentences)
        
        # Add category-specific prefix
        prefix = template['prompt'].replace(':', '').strip()
        return f"{prefix}: {summary}"
    
    def _post_process_summary(self, summary: str, query: str, category: str) -> str:
        """Post-process generated summary for clinical accuracy."""
        # Add clinical context if missing
        if category == 'dosing' and not any(term in summary.lower() for term in ['mg', 'dose', 'protocol']):
            summary = f"Dosing information: {summary}"
        elif category == 'toxicity' and not any(term in summary.lower() for term in ['toxicity', 'adverse', 'monitoring']):
            summary = f"Toxicity and monitoring: {summary}"
        
        # Clean up common issues
        summary = re.sub(r'\s+', ' ', summary)  # Multiple spaces
        summary = re.sub(r'\[.*?\]', '', summary)  # Remove source brackets
        summary = summary.strip()
        
        return summary
    
    def _extract_clinical_facts(self, content: str, query: str) -> List[str]:
        """Extract key clinical facts from content."""
        facts = []
        
        # Extract dosing information
        dosing_matches = self.clinical_extractors['dosing_values'].findall(content)
        for dose, unit in dosing_matches:
            facts.append(f"Dose: {dose} {unit}")
        
        # Extract toxicity grades
        toxicity_matches = self.clinical_extractors['toxicity_grades'].findall(content)
        for grade in toxicity_matches:
            facts.append(f"Toxicity: {grade}")
        
        # Extract lab values
        lab_matches = self.clinical_extractors['lab_values'].findall(content)
        for value, test in lab_matches:
            facts.append(f"Lab threshold: {value} {test}")
        
        # Extract frequencies
        freq_matches = self.clinical_extractors['frequencies'].findall(content)
        for freq in freq_matches:
            facts.append(f"Frequency: {freq}")
        
        # Extract clinical actions
        action_matches = self.clinical_extractors['clinical_actions'].findall(content)
        for action in action_matches:
            facts.append(f"Action: {action.capitalize()}")
        
        return list(set(facts))  # Remove duplicates
    
    def _calculate_confidence_score(self, documents: List[str], metadatas: List[Dict], query: str) -> float:
        """Calculate confidence score for the summary."""
        if not documents:
            return 0.0
        
        total_score = 0
        query_words = set(query.lower().split())
        
        for doc, metadata in zip(documents, metadatas if metadatas else [{}] * len(documents)):
            doc_score = 0
            doc_words = set(doc.lower().split())
            
            # Query overlap
            overlap = len(query_words.intersection(doc_words))
            doc_score += overlap * 0.1
            
            # Source credibility
            evidence_level = metadata.get('evidence_level', '').lower()
            if 'fda approved' in evidence_level:
                doc_score += 0.3
            elif 'clinical trial' in evidence_level:
                doc_score += 0.2
            elif 'guideline' in evidence_level:
                doc_score += 0.25
            
            # Document type credibility
            doc_type = metadata.get('doc_type', '').lower()
            if 'prescribing information' in doc_type:
                doc_score += 0.2
            elif 'clinical protocol' in doc_type:
                doc_score += 0.15
            
            total_score += doc_score
        
        # Normalize by number of documents
        confidence = min(1.0, total_score / len(documents))
        return round(confidence, 2)
    
    def _get_evidence_sources(self, metadatas: List[Dict]) -> List[str]:
        """Extract evidence sources from metadata."""
        sources = []
        for metadata in metadatas:
            source_info = []
            if metadata.get('source'):
                source_info.append(metadata['source'])
            if metadata.get('evidence_level'):
                source_info.append(f"({metadata['evidence_level']})")
            
            if source_info:
                sources.append(' '.join(source_info))
        
        return list(set(sources))  # Remove duplicates
    
    def _identify_clinical_warnings(self, content: str) -> List[str]:
        """Identify clinical warnings in content."""
        warnings = []
        content_lower = content.lower()
        
        # Common warning indicators
        warning_patterns = [
            (r'\bcontraindicated?\b', 'Contraindication identified'),
            (r'\bblack box warning\b', 'Black box warning present'),
            (r'\bdiscontinue\b', 'Discontinuation may be required'),
            (r'\bgrade\s+[34]\b', 'Severe toxicity (Grade 3-4) mentioned'),
            (r'\bfatal\b|\bdeath\b', 'Fatal outcomes mentioned'),
            (r'\bemergency\b|\burgent\b', 'Emergency situation noted')
        ]
        
        for pattern, warning in warning_patterns:
            if re.search(pattern, content_lower):
                warnings.append(warning)
        
        return warnings
    
    def format_clinical_summary(self, summary_result: Dict[str, Any]) -> str:
        """Format clinical summary for display."""
        output = []
        
        # Header
        output.append("📋 Clinical Summary")
        output.append("=" * 30)
        
        # Main summary
        output.append(f"🎯 {summary_result['summary']}")
        
        # Key facts
        if summary_result.get('key_facts'):
            output.append("\n📊 Key Clinical Facts:")
            for fact in summary_result['key_facts'][:5]:  # Top 5 facts
                output.append(f"  • {fact}")
        
        # Warnings
        if summary_result.get('warnings'):
            output.append("\n⚠️ Clinical Warnings:")
            for warning in summary_result['warnings']:
                output.append(f"  ⚠️ {warning}")
        
        # Evidence information
        confidence = summary_result.get('confidence_score', 0)
        output.append(f"\n📊 Confidence: {confidence:.0%}")
        output.append(f"📚 Based on {summary_result.get('source_count', 0)} sources")
        
        if summary_result.get('evidence_sources'):
            output.append("🏛️ Evidence sources:")
            for source in summary_result['evidence_sources'][:3]:  # Top 3 sources
                output.append(f"  • {source}")
        
        output.append(f"\n🔧 Generated using: {summary_result.get('generation_method', 'unknown')} method")
        
        return "\n".join(output)

def test_summarizer():
    """Test the clinical summarizer."""
    summarizer = ClinicalSummarizer()
    
    # Test query and mock results
    test_query = "temozolomide dosing newly diagnosed GBM"
    
    test_results = {
        'results': {
            'documents': [[
                "The standard temozolomide dose for newly diagnosed glioblastoma is 75 mg/m² daily during concurrent chemoradiation for 6 weeks, followed by maintenance therapy at 150-200 mg/m² for days 1-5 of each 28-day cycle for up to 6 cycles.",
                "Monitor CBC weekly during concurrent phase. For Grade 3 thrombocytopenia (platelets <50,000/μL), hold TMZ until recovery to Grade 1, then reduce dose by 25 mg/m².",
                "Temozolomide should be taken on an empty stomach. Common adverse effects include nausea, vomiting, fatigue, and myelosuppression."
            ]],
            'metadatas': [[
                {'source': 'FDA Prescribing Information', 'evidence_level': 'FDA approved', 'doc_type': 'prescribing_information'},
                {'source': 'NCCN Guidelines', 'evidence_level': 'clinical guideline', 'doc_type': 'clinical_protocol'},
                {'source': 'Clinical Trial Data', 'evidence_level': 'clinical trial', 'doc_type': 'research_article'}
            ]]
        }
    }
    
    # Generate summary
    summary_result = summarizer.summarize_clinical_results(test_query, test_results)
    
    # Format and display
    formatted_summary = summarizer.format_clinical_summary(summary_result)
    print("🧪 Testing Clinical Summarizer")
    print("=" * 50)
    print(formatted_summary)

if __name__ == "__main__":
    test_summarizer()