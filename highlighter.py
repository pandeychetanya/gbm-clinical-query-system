#!/usr/bin/env python3
"""
Snippet Highlighting Module for GBM Clinical Query Interface
Highlights key query terms and clinical keywords in retrieved text snippets
Author: Chetanya Pandey
"""

import re
from typing import List, Dict, Set, Tuple
import html

class ClinicalSnippetHighlighter:
    def __init__(self):
        """Initialize the clinical snippet highlighter."""
        
        # Clinical keywords that should always be highlighted
        self.clinical_keywords = {
            # Drug names
            'temozolomide', 'tmz', 'temodar', 'temodal',
            'bevacizumab', 'avastin', 'anti-vegf',
            
            # Dosing terms
            'mg/m²', 'mg/kg', 'mg/m2', 'dose', 'dosing', 'dosage',
            'protocol', 'regimen', 'schedule', 'administration',
            'concurrent', 'concomitant', 'maintenance', 'adjuvant',
            
            # Laboratory values
            'cbc', 'complete blood count', 'platelet', 'platelets',
            'neutrophil', 'neutrophils', 'anc', 'hemoglobin',
            'hematocrit', 'white blood cell', 'wbc',
            
            # Toxicity terms
            'thrombocytopenia', 'neutropenia', 'anemia', 'leukopenia',
            'toxicity', 'adverse', 'side effect', 'side effects',
            'grade 1', 'grade 2', 'grade 3', 'grade 4',
            'ctcae', 'common terminology criteria',
            
            # Clinical actions
            'hold', 'withhold', 'discontinue', 'stop', 'interrupt',
            'reduce', 'modify', 'adjust', 'decrease', 'increase',
            
            # Thresholds and values
            '150 mg/m²', '200 mg/m²', '75 mg/m²', '100 mg/m²',
            '5 mg/kg', '10 mg/kg', '15 mg/kg',
            '50,000', '75,000', '100,000', '1,000', '1,500',
            
            # Clinical conditions
            'glioblastoma', 'gbm', 'glioblastoma multiforme',
            'newly diagnosed', 'recurrent', 'progressive',
            'pseudoprogression', 'radiation necrosis',
            
            # Treatment phases
            'chemoradiation', 'chemoradiotherapy', 'radiotherapy',
            'stupp protocol', 'first-line', 'second-line',
            
            # Monitoring
            'monitoring', 'surveillance', 'follow-up', 'assessment',
            'laboratory', 'blood work', 'weekly', 'monthly',
            
            # Contraindications
            'contraindicated', 'contraindication', 'avoid', 'caution',
            'warning', 'precaution', 'not recommended',
            
            # Performance status
            'kps', 'karnofsky', 'ecog', 'performance status',
            'functional status', 'ps',
            
            # Molecular markers
            'mgmt', 'methylation', 'methylated', 'unmethylated',
            'o6-methylguanine-dna methyltransferase',
            'idh', 'idh1', 'idh2', '1p/19q',
            
            # Evidence levels
            'fda approved', 'phase i', 'phase ii', 'phase iii',
            'randomized', 'clinical trial', 'nccn', 'guideline'
        }
        
        # HTML/terminal color codes for different highlight types
        self.highlight_styles = {
            'query_match': {
                'html': '<span style="background-color: #ffff00; font-weight: bold; color: #000000;">',
                'terminal': '\033[1;43;30m',  # Bold yellow background, black text
                'markdown': '**',
                'end_html': '</span>',
                'end_terminal': '\033[0m',
                'end_markdown': '**'
            },
            'clinical_keyword': {
                'html': '<span style="background-color: #87ceeb; font-weight: bold; color: #000080;">',
                'terminal': '\033[1;46;34m',  # Bold cyan background, blue text
                'markdown': '*',
                'end_html': '</span>',
                'end_terminal': '\033[0m',
                'end_markdown': '*'
            },
            'dosage_value': {
                'html': '<span style="background-color: #98fb98; font-weight: bold; color: #006400;">',
                'terminal': '\033[1;42;32m',  # Bold green background, green text
                'markdown': '***',
                'end_html': '</span>',
                'end_terminal': '\033[0m',
                'end_markdown': '***'
            },
            'grade_toxicity': {
                'html': '<span style="background-color: #ffa07a; font-weight: bold; color: #8b0000;">',
                'terminal': '\033[1;41;31m',  # Bold red background, red text
                'markdown': '~~',
                'end_html': '</span>',
                'end_terminal': '\033[0m',
                'end_markdown': '~~'
            }
        }
        
        # Patterns for special highlighting
        self.dosage_pattern = re.compile(
            r'\b(\d+(?:\.\d+)?)\s*(mg/m²|mg/kg|mg/m2|mg)\b',
            re.IGNORECASE
        )
        
        self.grade_pattern = re.compile(
            r'\b(grade\s+[1-4]|ctc\s+grade\s+[1-4]|ctcae\s+grade\s+[1-4])\b',
            re.IGNORECASE
        )
        
        self.lab_value_pattern = re.compile(
            r'\b(\d+,?\d*)\s*(platelets?|neutrophils?|hemoglobin|hgb|hematocrit|hct|anc)(?:/μl|/mcl|/ml)?\b',
            re.IGNORECASE
        )
    
    def highlight_snippet(self, text: str, query: str, format_type: str = 'terminal', 
                         max_length: int = 500) -> str:
        """
        Highlight key terms in text snippet.
        
        Args:
            text: Text to highlight
            query: Original query terms
            format_type: 'terminal', 'html', or 'markdown'
            max_length: Maximum snippet length
            
        Returns:
            Highlighted text snippet
        """
        if not text or not text.strip():
            return text
        
        # Truncate text to manageable length while preserving word boundaries
        if len(text) > max_length:
            text = self._smart_truncate(text, query, max_length)
        
        # Extract query terms
        query_terms = self._extract_query_terms(query)
        
        # Create highlighting plan
        highlight_plan = self._create_highlight_plan(text, query_terms)
        
        # Apply highlights
        highlighted_text = self._apply_highlights(text, highlight_plan, format_type)
        
        return highlighted_text
    
    def _smart_truncate(self, text: str, query: str, max_length: int) -> str:
        """Intelligently truncate text to show most relevant content."""
        if len(text) <= max_length:
            return text
        
        # Find best snippet location based on query terms
        query_terms = self._extract_query_terms(query)
        best_start = 0
        max_matches = 0
        
        # Sliding window to find section with most query matches
        window_size = max_length - 100  # Leave room for context
        
        for start in range(0, len(text) - window_size, 50):
            snippet = text[start:start + window_size].lower()
            matches = sum(1 for term in query_terms if term.lower() in snippet)
            
            if matches > max_matches:
                max_matches = matches
                best_start = start
        
        # Adjust to word boundaries
        if best_start > 0:
            # Find previous sentence or paragraph break
            for i in range(best_start, max(0, best_start - 100), -1):
                if text[i] in '.!?\n':
                    best_start = i + 1
                    break
        
        # Extract snippet
        snippet = text[best_start:best_start + max_length]
        
        # Trim to last complete sentence if possible
        last_sentence = max(
            snippet.rfind('.'),
            snippet.rfind('!'),
            snippet.rfind('?')
        )
        
        if last_sentence > max_length * 0.7:  # Only if we don't lose too much
            snippet = snippet[:last_sentence + 1]
        
        # Add ellipsis
        prefix = "..." if best_start > 0 else ""
        suffix = "..." if best_start + len(snippet) < len(text) else ""
        
        return prefix + snippet.strip() + suffix
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query."""
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'what', 'when', 'where', 'why', 'how', 'who', 'which'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        meaningful_terms = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Add compound terms (phrases)
        phrases = []
        query_lower = query.lower()
        
        # Medical compound terms
        medical_phrases = [
            'mg/m²', 'mg/kg', 'grade 1', 'grade 2', 'grade 3', 'grade 4',
            'newly diagnosed', 'recurrent gbm', 'performance status',
            'complete blood count', 'absolute neutrophil count',
            'stupp protocol', 'maintenance therapy'
        ]
        
        for phrase in medical_phrases:
            if phrase in query_lower:
                phrases.append(phrase)
        
        return meaningful_terms + phrases
    
    def _create_highlight_plan(self, text: str, query_terms: List[str]) -> List[Tuple[int, int, str]]:
        """Create a plan of what to highlight where."""
        highlight_plan = []  # List of (start, end, highlight_type)
        text_lower = text.lower()
        
        # 1. Highlight query term matches (highest priority)
        for term in query_terms:
            term_lower = term.lower()
            start = 0
            while True:
                pos = text_lower.find(term_lower, start)
                if pos == -1:
                    break
                
                # Check word boundaries for single words
                if len(term.split()) == 1:
                    if pos > 0 and text_lower[pos-1].isalnum():
                        start = pos + 1
                        continue
                    if pos + len(term) < len(text_lower) and text_lower[pos + len(term)].isalnum():
                        start = pos + 1
                        continue
                
                highlight_plan.append((pos, pos + len(term), 'query_match'))
                start = pos + 1
        
        # 2. Highlight dosage values
        for match in self.dosage_pattern.finditer(text):
            start, end = match.span()
            highlight_plan.append((start, end, 'dosage_value'))
        
        # 3. Highlight toxicity grades
        for match in self.grade_pattern.finditer(text):
            start, end = match.span()
            highlight_plan.append((start, end, 'grade_toxicity'))
        
        # 4. Highlight clinical keywords (lower priority)
        for keyword in self.clinical_keywords:
            start = 0
            while True:
                pos = text_lower.find(keyword.lower(), start)
                if pos == -1:
                    break
                
                # Check word boundaries
                if pos > 0 and text_lower[pos-1].isalnum():
                    start = pos + 1
                    continue
                if pos + len(keyword) < len(text_lower) and text_lower[pos + len(keyword)].isalnum():
                    start = pos + 1
                    continue
                
                highlight_plan.append((pos, pos + len(keyword), 'clinical_keyword'))
                start = pos + 1
        
        # Sort by position and resolve overlaps (higher priority wins)
        highlight_plan.sort(key=lambda x: (x[0], -self._get_priority(x[2])))
        
        # Remove overlaps
        filtered_plan = []
        for start, end, highlight_type in highlight_plan:
            # Check for overlap with existing highlights
            overlap = False
            for existing_start, existing_end, existing_type in filtered_plan:
                if not (end <= existing_start or start >= existing_end):
                    # There's an overlap
                    if self._get_priority(highlight_type) > self._get_priority(existing_type):
                        # Remove the existing highlight
                        filtered_plan.remove((existing_start, existing_end, existing_type))
                        break
                    else:
                        overlap = True
                        break
            
            if not overlap:
                filtered_plan.append((start, end, highlight_type))
        
        return sorted(filtered_plan)
    
    def _get_priority(self, highlight_type: str) -> int:
        """Get priority for highlight type (higher number = higher priority)."""
        priorities = {
            'query_match': 4,
            'dosage_value': 3,
            'grade_toxicity': 3,
            'clinical_keyword': 1
        }
        return priorities.get(highlight_type, 0)
    
    def _apply_highlights(self, text: str, highlight_plan: List[Tuple[int, int, str]], 
                         format_type: str) -> str:
        """Apply highlighting to text based on plan."""
        if not highlight_plan:
            return text
        
        result = []
        last_pos = 0
        
        for start, end, highlight_type in highlight_plan:
            # Add text before highlight
            result.append(text[last_pos:start])
            
            # Add highlighted text
            style = self.highlight_styles[highlight_type]
            highlighted_text = text[start:end]
            
            if format_type == 'html':
                result.append(f"{style['html']}{html.escape(highlighted_text)}{style['end_html']}")
            elif format_type == 'terminal':
                result.append(f"{style['terminal']}{highlighted_text}{style['end_terminal']}")
            elif format_type == 'markdown':
                result.append(f"{style['markdown']}{highlighted_text}{style['end_markdown']}")
            else:
                result.append(highlighted_text)
            
            last_pos = end
        
        # Add remaining text
        result.append(text[last_pos:])
        
        return ''.join(result)
    
    def create_legend(self, format_type: str = 'terminal') -> str:
        """Create a legend explaining the highlight colors."""
        if format_type == 'terminal':
            return f"""
🌈 Highlight Legend:
{self.highlight_styles['query_match']['terminal']}Query Matches{self.highlight_styles['query_match']['end_terminal']} - Your search terms
{self.highlight_styles['clinical_keyword']['terminal']}Clinical Keywords{self.highlight_styles['clinical_keyword']['end_terminal']} - Important medical terms
{self.highlight_styles['dosage_value']['terminal']}Dosage Values{self.highlight_styles['dosage_value']['end_terminal']} - Drug doses and measurements
{self.highlight_styles['grade_toxicity']['terminal']}Toxicity Grades{self.highlight_styles['grade_toxicity']['end_terminal']} - Severity classifications
"""
        elif format_type == 'html':
            return f"""
<div style="margin: 10px 0; padding: 10px; border: 1px solid #ccc; background-color: #f9f9f9;">
<h4>🌈 Highlight Legend:</h4>
<p>{self.highlight_styles['query_match']['html']}Query Matches{self.highlight_styles['query_match']['end_html']} - Your search terms</p>
<p>{self.highlight_styles['clinical_keyword']['html']}Clinical Keywords{self.highlight_styles['clinical_keyword']['end_html']} - Important medical terms</p>
<p>{self.highlight_styles['dosage_value']['html']}Dosage Values{self.highlight_styles['dosage_value']['end_html']} - Drug doses and measurements</p>
<p>{self.highlight_styles['grade_toxicity']['html']}Toxicity Grades{self.highlight_styles['grade_toxicity']['end_html']} - Severity classifications</p>
</div>
"""
        else:  # markdown
            return """
🌈 **Highlight Legend:**
- **Query Matches** - Your search terms
- *Clinical Keywords* - Important medical terms  
- ***Dosage Values*** - Drug doses and measurements
- ~~Toxicity Grades~~ - Severity classifications
"""
    
    def highlight_multiple_snippets(self, snippets: List[Dict], query: str, 
                                   format_type: str = 'terminal') -> List[Dict]:
        """Highlight multiple snippets with metadata."""
        highlighted_snippets = []
        
        for snippet_data in snippets:
            if isinstance(snippet_data, dict) and 'text' in snippet_data:
                highlighted_text = self.highlight_snippet(
                    snippet_data['text'], query, format_type
                )
                
                highlighted_snippet = snippet_data.copy()
                highlighted_snippet['highlighted_text'] = highlighted_text
                highlighted_snippet['original_text'] = snippet_data['text']
                highlighted_snippets.append(highlighted_snippet)
            else:
                # Handle simple text snippets
                text = snippet_data if isinstance(snippet_data, str) else str(snippet_data)
                highlighted_text = self.highlight_snippet(text, query, format_type)
                
                highlighted_snippets.append({
                    'text': text,
                    'highlighted_text': highlighted_text,
                    'original_text': text
                })
        
        return highlighted_snippets

def test_highlighter():
    """Test the highlighting functionality."""
    highlighter = ClinicalSnippetHighlighter()
    
    test_cases = [
        {
            'query': 'temozolomide dosing newly diagnosed GBM',
            'text': 'The standard temozolomide dose for newly diagnosed glioblastoma is 75 mg/m² daily during concurrent chemoradiation, followed by maintenance therapy at 150-200 mg/m² for days 1-5 of each 28-day cycle. Monitor CBC weekly during concurrent phase and before each cycle during maintenance.'
        },
        {
            'query': 'bevacizumab side effects monitoring',
            'text': 'Bevacizumab adverse effects include arterial thrombotic events, hemorrhage, and proteinuria. Monitor blood pressure before each dose. Obtain urinalysis for proteinuria. Hold bevacizumab for Grade 2 or higher proteinuria (≥2+ protein or >100 mg/dL). Discontinue for Grade 4 hemorrhage.'
        },
        {
            'query': 'TMZ dose modification thrombocytopenia',
            'text': 'For Grade 3 thrombocytopenia (platelets <50,000/μL), hold TMZ until recovery to Grade 1, then reduce dose by 25 mg/m². For Grade 4 thrombocytopenia (platelets <25,000/μL), discontinue TMZ permanently. Platelet count must be ≥100,000/μL before starting each cycle.'
        }
    ]
    
    print("🧪 Testing Clinical Snippet Highlighter")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        print(f"Query: {test_case['query']}")
        print(f"Original: {test_case['text'][:100]}...")
        
        highlighted = highlighter.highlight_snippet(
            test_case['text'], 
            test_case['query'], 
            format_type='terminal'
        )
        
        print(f"Highlighted:")
        print(highlighted)
        print("-" * 40)
    
    # Show legend
    print(highlighter.create_legend('terminal'))

if __name__ == "__main__":
    test_highlighter()