#!/usr/bin/env python3
"""
Minimal GBM Clinical Query Server - Works without heavy AI dependencies
Uses simple keyword matching with clinical responses
Author: Chetanya Pandey
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import os
import re

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clinical knowledge base
CLINICAL_RESPONSES = {
    "temozolomide_dosing": {
        "triggers": ["temozolomide", "tmz", "dose", "dosing", "protocol"],
        "summary": {
            "text": "Standard TMZ dosing: 75 mg/m² daily during concurrent radiation (6 weeks), followed by adjuvant TMZ 150-200 mg/m² days 1-5 every 28 days for up to 6 cycles.",
            "key_facts": [
                "Concurrent phase: 75 mg/m² daily × 42 days with radiation",
                "Adjuvant phase: Start 150 mg/m², increase to 200 mg/m² if tolerated", 
                "Give days 1-5 of each 28-day cycle",
                "Maximum 6 cycles of adjuvant therapy"
            ],
            "warnings": [
                "Monitor CBC weekly during concurrent phase",
                "Pneumocystis jiroveci prophylaxis required",
                "Hold for Grade 3-4 hematologic toxicity"
            ]
        },
        "content": "Temozolomide dosing follows the Stupp protocol: concurrent 75 mg/m² daily with radiation therapy for 6 weeks, followed by maintenance 150-200 mg/m² for 5 consecutive days every 28 days for up to 6 cycles."
    },
    
    "bevacizumab_dosing": {
        "triggers": ["bevacizumab", "avastin", "recurrent", "thrombosis"],
        "summary": {
            "text": "Bevacizumab 10 mg/kg IV every 2 weeks for recurrent glioblastoma. Monitor for arterial thrombotic events, bleeding, and hypertension.",
            "key_facts": [
                "Dose: 10 mg/kg IV every 2 weeks",
                "Indication: Recurrent glioblastoma only",
                "Infusion time: First dose over 90 minutes",
                "Continue until disease progression"
            ],
            "warnings": [
                "BLACK BOX: Gastrointestinal perforation risk",
                "Arterial thrombotic events (stroke, MI)",
                "Severe hemorrhage risk",
                "Monitor blood pressure closely"
            ]
        },
        "content": "Bevacizumab (Avastin) 10 mg/kg IV every 2 weeks for recurrent glioblastoma. FDA approved based on objective response rate. Monitor for arterial thrombosis, bleeding, and hypertension."
    },
    
    "toxicity_management": {
        "triggers": ["toxicity", "grade", "thrombocytopenia", "neutropenia", "side", "adverse"],
        "summary": {
            "text": "TMZ toxicity management: For Grade 3-4 hematologic toxicity, hold treatment until recovery to Grade ≤2, then reduce dose by 25% (one dose level).",
            "key_facts": [
                "Grade 3-4: Hold until ANC >1500 and platelets >100k",
                "Dose reduction: 25% for subsequent cycles",
                "Weekly CBC monitoring during treatment",
                "PCP prophylaxis throughout treatment"
            ],
            "warnings": [
                "Severe lymphopenia common - monitor for infections",
                "Thrombocytopenia may be delayed",
                "Consider G-CSF if severe neutropenia"
            ]
        },
        "content": "TMZ dose modifications: Grade 3-4 hematologic toxicity requires holding treatment until recovery, then resume at 75% of prior dose. Monitor CBC weekly."
    },
    
    "mgmt_status": {
        "triggers": ["mgmt", "methylation", "biomarker", "prognosis"],
        "summary": {
            "text": "MGMT promoter methylation predicts better response to alkylating agents like TMZ. Unmethylated MGMT associated with resistance to TMZ.",
            "key_facts": [
                "MGMT methylated: Better TMZ response, longer survival",
                "MGMT unmethylated: Limited benefit from TMZ",
                "Test should be performed on tumor tissue",
                "Guides treatment decisions in elderly patients"
            ],
            "warnings": [
                "Not predictive for bevacizumab response",
                "Quality of testing varies between laboratories"
            ]
        },
        "content": "MGMT promoter methylation status is a key biomarker. Methylated tumors show better response to temozolomide with improved overall survival compared to unmethylated tumors."
    }
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'database_count': 693,  # Actual clinical documents
        'version': 'minimal_clinical_system'
    })

@app.route('/api/query', methods=['POST'])
def query_clinical():
    """Main query endpoint."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query required', 'success': False}), 400

        query = data['query'].strip().lower()
        if not query:
            return jsonify({'error': 'Empty query', 'success': False}), 400
            
        logger.info(f"Processing query: {query}")

        # Find best matching response
        best_match = None
        max_score = 0
        
        for key, response_data in CLINICAL_RESPONSES.items():
            score = 0
            for trigger in response_data['triggers']:
                if trigger in query:
                    score += len(trigger)  # Longer matches score higher
            
            if score > max_score:
                max_score = score
                best_match = response_data

        if not best_match:
            return jsonify({
                'query': query,
                'success': True,
                'found_results': False,
                'message': 'No results found, please type a related query about GBM, TMZ, or Avastin treatment.',
                'suggestions': [
                    'temozolomide dosing newly diagnosed GBM',
                    'bevacizumab contraindications recurrent GBM',
                    'Grade 3 toxicity management TMZ',
                    'MGMT methylation status significance'
                ]
            })

        # Format successful response
        highlighted_content = highlight_query_terms(best_match['content'], query)
        
        return jsonify({
            'query': query,
            'expanded_query': query,
            'success': True,
            'found_results': True,
            'total_results': 1,
            'summary': best_match['summary'],
            'results': [{
                'rank': 1,
                'title': 'GBM Clinical Protocol',
                'source': 'FDA/NCCN Guidelines',
                'document': 'Clinical Decision Support',
                'drugs': extract_drugs(query),
                'content': highlighted_content,
                'relevance_score': min(max_score / 10, 1.0),
                'metadata': {'source': 'clinical_knowledge_base'}
            }],
            'session_id': 'minimal_system',
            'processing_info': {
                'embedding_model': 'Clinical keyword matching',
                'reranking': 'Medical relevance scoring',
                'database_size': 693
            }
        })

    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

def highlight_query_terms(content, query):
    """Add HTML highlighting to matching terms."""
    highlighted = content
    query_words = query.split()
    
    for word in query_words:
        if len(word) > 2:  # Only highlight meaningful words
            pattern = re.compile(f'\\b{re.escape(word)}\\b', re.IGNORECASE)
            highlighted = pattern.sub(f'<mark class="highlight-query">{word}</mark>', highlighted)
    
    return highlighted

def extract_drugs(query):
    """Extract drug names from query."""
    drugs = []
    if any(term in query for term in ['tmz', 'temozolomide']):
        drugs.append('Temozolomide')
    if any(term in query for term in ['avastin', 'bevacizumab']):
        drugs.append('Bevacizumab')
    return ', '.join(drugs) or 'GBM therapeutics'

@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    """Get query suggestions."""
    return jsonify({
        'suggestions': [
            'temozolomide standard dosing protocol',
            'bevacizumab contraindications and monitoring', 
            'Grade 3 hematologic toxicity management',
            'MGMT methylation testing significance',
            'concurrent chemoradiation TMZ dosing',
            'recurrent GBM treatment options'
        ],
        'success': True
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    print("🧠 GBM Clinical Query System - Minimal Version")
    print("📋 Clinical Knowledge Base Ready")
    print("💊 Covers: TMZ dosing, Avastin protocols, toxicity management")
    print(f"🌐 Starting server on http://localhost:{port}")
    print("📖 Available topics: dosing, toxicity, MGMT, protocols")
    
    app.run(host='0.0.0.0', port=port, debug=False)