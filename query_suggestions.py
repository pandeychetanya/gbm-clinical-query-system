#!/usr/bin/env python3
"""
Alternative Query Formulation System for GBM Clinical Query Interface
Provides intelligent query suggestions to help clinicians refine and improve their searches
Author: Chetanya Pandey
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from difflib import SequenceMatcher
import json

class ClinicalQuerySuggestions:
    def __init__(self):
        """Initialize the clinical query suggestion system."""
        
        # Clinical query templates organized by intent
        self.query_templates = {
            'dosing': {
                'patterns': ['dose', 'dosing', 'mg/m²', 'protocol', 'regimen', 'administration'],
                'templates': [
                    "What is the standard {drug} dose for {population}?",
                    "How is {drug} dosed in {population}?", 
                    "{drug} dosing protocol for {indication}",
                    "Standard {drug} regimen for {phase} treatment",
                    "{drug} mg/m² dosing schedule",
                    "How to administer {drug} for {indication}?",
                    "{drug} dose modification guidelines",
                    "Concurrent {drug} dosing during {treatment}"
                ]
            },
            'toxicity': {
                'patterns': ['toxicity', 'side effects', 'adverse', 'monitoring', 'safety'],
                'templates': [
                    "What are the side effects of {drug}?",
                    "{drug} toxicity monitoring guidelines",
                    "How to monitor {drug} safety?",
                    "{drug} adverse effects in {population}",
                    "Laboratory monitoring for {drug}",
                    "{drug} dose modifications for toxicity",
                    "Managing {drug} side effects",
                    "{drug} contraindications and warnings"
                ]
            },
            'administration': {
                'patterns': ['give', 'administer', 'infusion', 'preparation', 'timing'],
                'templates': [
                    "How to prepare {drug} for administration?",
                    "{drug} infusion protocol",
                    "When to give {drug}?",
                    "{drug} administration guidelines",
                    "Premedications for {drug}",
                    "{drug} timing with other treatments",
                    "How to mix {drug}?",
                    "{drug} storage and handling"
                ]
            },
            'contraindications': {
                'patterns': ['avoid', 'contraindication', 'warning', 'precaution', 'caution'],
                'templates': [
                    "When is {drug} contraindicated?",
                    "Who should avoid {drug}?",
                    "{drug} warnings and precautions",
                    "{drug} contraindications in {population}",
                    "When not to use {drug}?",
                    "{drug} safety considerations",
                    "{drug} black box warnings",
                    "Absolute contraindications for {drug}"
                ]
            },
            'interactions': {
                'patterns': ['interaction', 'concurrent', 'combination', 'together'],
                'templates': [
                    "{drug} drug interactions",
                    "Can {drug} be given with {other_drug}?",
                    "{drug} and {treatment} combination",
                    "Concurrent {drug} and radiation",
                    "{drug} interaction warnings",
                    "What drugs interact with {drug}?",
                    "{drug} combination protocols",
                    "{drug} with other chemotherapy"
                ]
            },
            'monitoring': {
                'patterns': ['monitor', 'follow-up', 'surveillance', 'lab', 'CBC'],
                'templates': [
                    "How to monitor patients on {drug}?",
                    "{drug} laboratory monitoring schedule",
                    "What labs to check with {drug}?",
                    "{drug} surveillance requirements",
                    "Monitoring for {drug} toxicity",
                    "{drug} follow-up protocols",
                    "CBC monitoring during {drug}",
                    "When to check labs on {drug}?"
                ]
            },
            'efficacy': {
                'patterns': ['efficacy', 'effectiveness', 'survival', 'outcome', 'response'],
                'templates': [
                    "How effective is {drug} for {indication}?",
                    "{drug} survival benefits in {population}",
                    "{drug} response rates for {indication}",
                    "Clinical outcomes with {drug}",
                    "{drug} efficacy data",
                    "Survival data for {drug} in {indication}",
                    "{drug} treatment outcomes",
                    "Evidence for {drug} in {population}"
                ]
            }
        }
        
        # Drug name mappings and synonyms
        self.drug_synonyms = {
            'temozolomide': ['temozolomide', 'tmz', 'temodar', 'temodal'],
            'bevacizumab': ['bevacizumab', 'avastin', 'anti-vegf'],
            'lomustine': ['lomustine', 'ccnu', 'ceenu'],
            'carmustine': ['carmustine', 'bcnu', 'bischloroethylnitrosourea']
        }
        
        # Clinical populations and conditions
        self.populations = [
            'newly diagnosed GBM', 'recurrent GBM', 'progressive GBM',
            'elderly patients', 'pediatric patients', 'young adults',
            'poor performance status patients', 'good performance status patients',
            'MGMT methylated patients', 'MGMT unmethylated patients'
        ]
        
        # Treatment phases
        self.treatment_phases = [
            'first-line', 'second-line', 'salvage',
            'concurrent', 'maintenance', 'adjuvant',
            'upfront', 'initial', 'subsequent'
        ]
        
        # Clinical indications
        self.indications = [
            'glioblastoma', 'GBM', 'anaplastic glioma',
            'brain tumor', 'malignant glioma', 'high-grade glioma'
        ]
        
        # Common query improvement patterns
        self.improvement_patterns = [
            {
                'pattern': r'\bTMZ\b',
                'suggestion': 'temozolomide',
                'reason': 'Use full drug name for more comprehensive results'
            },
            {
                'pattern': r'\bdose\b(?!\s+(reduction|modification|adjustment))',
                'suggestion': 'dosing protocol',
                'reason': 'More specific dosing terminology'
            },
            {
                'pattern': r'\bside effects?\b',
                'suggestion': 'adverse effects monitoring',
                'reason': 'Include monitoring for clinical context'
            },
            {
                'pattern': r'\bGBM\b(?!\s+(patients?|treatment))',
                'suggestion': 'GBM patients',
                'reason': 'Specify patient population'
            }
        ]
    
    def generate_alternative_queries(self, original_query: str, max_suggestions: int = 8) -> List[Dict[str, Any]]:
        """Generate alternative query formulations based on the original query."""
        suggestions = []
        
        # Detect query intent and extract entities
        intent = self._detect_query_intent(original_query)
        entities = self._extract_entities(original_query)
        
        # Generate template-based alternatives
        if intent in self.query_templates:
            template_suggestions = self._generate_template_suggestions(
                original_query, intent, entities
            )
            suggestions.extend(template_suggestions)
        
        # Generate refinement suggestions
        refinement_suggestions = self._generate_refinement_suggestions(original_query)
        suggestions.extend(refinement_suggestions)
        
        # Generate expansion suggestions
        expansion_suggestions = self._generate_expansion_suggestions(original_query, entities)
        suggestions.extend(expansion_suggestions)
        
        # Generate narrowing suggestions
        narrowing_suggestions = self._generate_narrowing_suggestions(original_query, entities)
        suggestions.extend(narrowing_suggestions)
        
        # Remove duplicates and rank by relevance
        unique_suggestions = self._deduplicate_suggestions(suggestions)
        ranked_suggestions = self._rank_suggestions(original_query, unique_suggestions)
        
        return ranked_suggestions[:max_suggestions]
    
    def _detect_query_intent(self, query: str) -> str:
        """Detect the primary intent of the query."""
        query_lower = query.lower()
        
        # Score each intent based on pattern matches
        intent_scores = {}
        
        for intent, config in self.query_templates.items():
            score = 0
            for pattern in config['patterns']:
                if pattern in query_lower:
                    score += 1
            intent_scores[intent] = score
        
        # Return the highest scoring intent, or 'general' if no clear winner
        if intent_scores:
            max_score = max(intent_scores.values())
            if max_score > 0:
                return max(intent_scores, key=intent_scores.get)
        
        return 'general'
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract clinical entities from the query."""
        entities = {
            'drugs': [],
            'populations': [],
            'indications': [],
            'phases': []
        }
        
        query_lower = query.lower()
        
        # Extract drugs
        for main_drug, synonyms in self.drug_synonyms.items():
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    entities['drugs'].append(main_drug)
                    break
        
        # Extract populations
        for population in self.populations:
            if population.lower() in query_lower:
                entities['populations'].append(population)
        
        # Extract treatment phases
        for phase in self.treatment_phases:
            if phase.lower() in query_lower:
                entities['phases'].append(phase)
        
        # Extract indications
        for indication in self.indications:
            if indication.lower() in query_lower:
                entities['indications'].append(indication)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _generate_template_suggestions(self, original_query: str, intent: str, 
                                     entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate suggestions based on query templates."""
        suggestions = []
        templates = self.query_templates[intent]['templates']
        
        # Default entities if none found
        default_drug = entities['drugs'][0] if entities['drugs'] else 'temozolomide'
        default_population = entities['populations'][0] if entities['populations'] else 'GBM patients'
        default_indication = entities['indications'][0] if entities['indications'] else 'GBM'
        default_phase = entities['phases'][0] if entities['phases'] else 'first-line'
        
        # Generate suggestions from templates
        for template in templates[:4]:  # Use first 4 templates
            try:
                suggestion = template.format(
                    drug=default_drug,
                    population=default_population,
                    indication=default_indication,
                    phase=default_phase,
                    treatment='radiation therapy',
                    other_drug='radiation'
                )
                
                suggestions.append({
                    'query': suggestion,
                    'type': f'{intent}_template',
                    'reason': f'Alternative {intent} query formulation',
                    'confidence': 0.8
                })
            except KeyError:
                # Template has placeholders we don't have entities for
                continue
        
        return suggestions
    
    def _generate_refinement_suggestions(self, original_query: str) -> List[Dict[str, Any]]:
        """Generate suggestions that refine the original query."""
        suggestions = []
        
        for pattern_config in self.improvement_patterns:
            if re.search(pattern_config['pattern'], original_query, re.IGNORECASE):
                refined_query = re.sub(
                    pattern_config['pattern'],
                    pattern_config['suggestion'],
                    original_query,
                    flags=re.IGNORECASE
                )
                
                if refined_query != original_query:
                    suggestions.append({
                        'query': refined_query,
                        'type': 'refinement',
                        'reason': pattern_config['reason'],
                        'confidence': 0.9
                    })
        
        return suggestions
    
    def _generate_expansion_suggestions(self, original_query: str, 
                                      entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate broader query suggestions."""
        suggestions = []
        
        # Add population specificity if missing
        if not entities['populations'] and not any(pop.lower() in original_query.lower() 
                                                 for pop in ['newly diagnosed', 'recurrent', 'elderly']):
            expanded_queries = [
                f"{original_query} in newly diagnosed patients",
                f"{original_query} for recurrent disease",
                f"{original_query} in elderly patients"
            ]
            
            for expanded in expanded_queries:
                suggestions.append({
                    'query': expanded,
                    'type': 'expansion',
                    'reason': 'Add patient population specificity',
                    'confidence': 0.7
                })
        
        # Add treatment context if missing
        if not any(context in original_query.lower() 
                  for context in ['concurrent', 'maintenance', 'first-line', 'adjuvant']):
            context_queries = [
                f"{original_query} concurrent with radiation",
                f"{original_query} maintenance therapy",
                f"{original_query} first-line treatment"
            ]
            
            for context_query in context_queries:
                suggestions.append({
                    'query': context_query,
                    'type': 'expansion',
                    'reason': 'Add treatment context',
                    'confidence': 0.6
                })
        
        return suggestions
    
    def _generate_narrowing_suggestions(self, original_query: str, 
                                      entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate more specific query suggestions."""
        suggestions = []
        
        # Add specific clinical focus
        if entities['drugs']:
            drug = entities['drugs'][0]
            specific_queries = [
                f"{drug} dose modification guidelines",
                f"{drug} FDA prescribing information",
                f"{drug} clinical trial results",
                f"{drug} safety monitoring protocol"
            ]
            
            for specific_query in specific_queries:
                if not self._queries_similar(original_query, specific_query):
                    suggestions.append({
                        'query': specific_query,
                        'type': 'narrowing',
                        'reason': f'Focus on specific {drug} information',
                        'confidence': 0.8
                    })
        
        # Add grade/severity specificity for toxicity queries
        if any(term in original_query.lower() for term in ['toxicity', 'side effects', 'adverse']):
            grade_queries = [
                f"{original_query} Grade 3 management",
                f"{original_query} severe complications",
                f"{original_query} dose-limiting toxicity"
            ]
            
            for grade_query in grade_queries:
                suggestions.append({
                    'query': grade_query,
                    'type': 'narrowing',
                    'reason': 'Add severity/grade specificity',
                    'confidence': 0.7
                })
        
        return suggestions
    
    def _queries_similar(self, query1: str, query2: str, threshold: float = 0.8) -> bool:
        """Check if two queries are too similar."""
        similarity = SequenceMatcher(None, query1.lower(), query2.lower()).ratio()
        return similarity > threshold
    
    def _deduplicate_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate and very similar suggestions."""
        unique_suggestions = []
        
        for suggestion in suggestions:
            is_unique = True
            for existing in unique_suggestions:
                if self._queries_similar(suggestion['query'], existing['query'], threshold=0.85):
                    is_unique = False
                    # Keep the one with higher confidence
                    if suggestion['confidence'] > existing['confidence']:
                        unique_suggestions.remove(existing)
                        is_unique = True
                    break
            
            if is_unique:
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def _rank_suggestions(self, original_query: str, 
                         suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank suggestions by relevance and quality."""
        
        # Calculate ranking score
        for suggestion in suggestions:
            score = suggestion['confidence']
            
            # Bonus for different suggestion types
            type_bonuses = {
                'refinement': 0.3,
                'narrowing': 0.2,
                'expansion': 0.1,
                'dosing_template': 0.25,
                'toxicity_template': 0.25
            }
            
            score += type_bonuses.get(suggestion['type'], 0)
            
            # Penalty for very long queries
            if len(suggestion['query']) > 100:
                score -= 0.1
            
            # Bonus for including specific clinical terms
            clinical_terms = ['protocol', 'guideline', 'monitoring', 'FDA', 'clinical trial']
            for term in clinical_terms:
                if term.lower() in suggestion['query'].lower():
                    score += 0.05
            
            suggestion['score'] = score
        
        # Sort by score (descending)
        return sorted(suggestions, key=lambda x: x['score'], reverse=True)
    
    def suggest_query_improvements(self, query: str) -> List[Dict[str, str]]:
        """Suggest specific improvements to the current query."""
        improvements = []
        query_lower = query.lower()
        
        # Suggest more specific drug names
        if re.search(r'\btmz\b', query_lower):
            improvements.append({
                'type': 'specificity',
                'suggestion': 'Use "temozolomide" instead of "TMZ"',
                'reason': 'Full drug names often return more comprehensive results'
            })
        
        # Suggest adding population context
        if not any(pop in query_lower for pop in ['newly diagnosed', 'recurrent', 'elderly', 'pediatric']):
            improvements.append({
                'type': 'context',
                'suggestion': 'Add patient population (e.g., "newly diagnosed", "recurrent")',
                'reason': 'Treatment varies by patient population and disease stage'
            })
        
        # Suggest adding clinical context
        if any(term in query_lower for term in ['dose', 'dosing']) and \
           not any(context in query_lower for context in ['concurrent', 'maintenance', 'adjuvant']):
            improvements.append({
                'type': 'context',
                'suggestion': 'Specify treatment phase (e.g., "concurrent", "maintenance")',
                'reason': 'Dosing protocols vary by treatment phase'
            })
        
        # Suggest evidence level specification
        if not any(evidence in query_lower for evidence in ['fda', 'guideline', 'protocol', 'trial']):
            improvements.append({
                'type': 'evidence',
                'suggestion': 'Add evidence source preference (e.g., "FDA approved", "NCCN guideline")',
                'reason': 'Helps filter for authoritative clinical information'
            })
        
        return improvements
    
    def format_suggestions(self, original_query: str, suggestions: List[Dict[str, Any]]) -> str:
        """Format query suggestions for display."""
        if not suggestions:
            return "No alternative query suggestions available."
        
        output = []
        output.append(f"💡 Alternative Query Suggestions for: '{original_query}'")
        output.append("=" * 60)
        
        # Group suggestions by type
        suggestion_groups = {}
        for suggestion in suggestions:
            group = suggestion['type'].split('_')[0]  # Get base type
            if group not in suggestion_groups:
                suggestion_groups[group] = []
            suggestion_groups[group].append(suggestion)
        
        # Display suggestions by group
        type_labels = {
            'refinement': '🔧 Query Refinements',
            'expansion': '🔍 Broader Searches',
            'narrowing': '🎯 More Specific Searches', 
            'dosing': '💊 Dosing Queries',
            'toxicity': '⚠️ Toxicity & Safety Queries',
            'administration': '📋 Administration Queries',
            'template': '📝 Alternative Formulations'
        }
        
        suggestion_count = 1
        for group, group_suggestions in suggestion_groups.items():
            if group_suggestions:
                label = type_labels.get(group, f'{group.title()} Suggestions')
                output.append(f"\n{label}:")
                
                for suggestion in group_suggestions[:3]:  # Max 3 per group
                    output.append(f"  {suggestion_count}. {suggestion['query']}")
                    output.append(f"     💡 {suggestion['reason']}")
                    suggestion_count += 1
        
        return "\n".join(output)

def test_query_suggestions():
    """Test the query suggestion system."""
    suggestion_system = ClinicalQuerySuggestions()
    
    test_queries = [
        "TMZ dose",
        "bevacizumab side effects",
        "GBM treatment protocol",
        "temozolomide monitoring",
        "avastin contraindications"
    ]
    
    print("🧪 Testing Clinical Query Suggestion System")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Original Query: '{query}'")
        
        # Generate alternative queries
        alternatives = suggestion_system.generate_alternative_queries(query)
        
        # Show formatted suggestions
        formatted = suggestion_system.format_suggestions(query, alternatives)
        print(formatted)
        
        # Show query improvements
        improvements = suggestion_system.suggest_query_improvements(query)
        if improvements:
            print(f"\n🔧 Query Improvement Tips:")
            for improvement in improvements:
                print(f"  • {improvement['suggestion']} - {improvement['reason']}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_query_suggestions()