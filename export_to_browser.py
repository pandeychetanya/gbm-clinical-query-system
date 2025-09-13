#!/usr/bin/env python3
"""
Export GBM Vector Database to Browser-Compatible Format
Converts ChromaDB to JSON for client-side processing
Author: Chetanya Pandey
"""

import json
import numpy as np
from pathlib import Path
import chromadb
from chromadb.config import Settings

def export_vector_db_to_json():
    """Export vector database to JSON format for browser."""
    
    print("🔄 Exporting vector database to browser format...")
    
    try:
        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(
            path="vector_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get collection
        try:
            collection = chroma_client.get_collection("gbm_clinical_medical_embeddings")
        except:
            try:
                collection = chroma_client.get_collection("gbm_clinical_data")
            except:
                print("❌ No valid collection found")
                return
        
        print(f"📊 Found {collection.count()} documents")
        
        # Get all documents
        results = collection.get(
            include=['documents', 'metadatas', 'embeddings']
        )
        
        # Convert to browser format
        browser_data = {
            'documents': [],
            'metadata': {
                'total_documents': len(results['documents']),
                'export_date': str(Path().resolve()),
                'model_info': 'Exported from GBM Clinical Query System'
            }
        }
        
        # Process each document
        for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            if len(doc.strip()) < 50:  # Skip very short documents
                continue
                
            # Create browser-compatible document
            browser_doc = {
                'id': i,
                'content': doc,
                'source': metadata.get('source', 'Unknown'),
                'filename': metadata.get('filename', f'doc_{i}'),
                'drugs': metadata.get('drugs', 'Unknown'),
                'topic': metadata.get('topic', 'General'),
                'keywords': extract_keywords(doc),
                'length': len(doc)
            }
            
            browser_data['documents'].append(browser_doc)
        
        # Save to JSON
        output_file = 'docs/clinical_database.json'
        with open(output_file, 'w') as f:
            json.dump(browser_data, f, indent=2)
        
        print(f"✅ Exported {len(browser_data['documents'])} documents to {output_file}")
        print(f"📁 File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
        
        # Create index for faster search
        create_search_index(browser_data)
        
        return browser_data
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return None

def extract_keywords(text):
    """Extract key medical terms from text."""
    keywords = []
    
    # Medical terms to look for
    medical_terms = [
        'temozolomide', 'tmz', 'bevacizumab', 'avastin',
        'glioblastoma', 'gbm', 'dose', 'dosing', 'mg/m2',
        'toxicity', 'grade', 'thrombocytopenia', 'neutropenia',
        'concurrent', 'adjuvant', 'maintenance', 'cycle',
        'radiation', 'chemoradiation', 'protocol'
    ]
    
    text_lower = text.lower()
    for term in medical_terms:
        if term in text_lower:
            keywords.append(term)
    
    return list(set(keywords))  # Remove duplicates

def create_search_index(data):
    """Create search index for faster browser queries."""
    
    search_index = {
        'keywords': {},
        'drugs': {},
        'sources': {}
    }
    
    for doc in data['documents']:
        doc_id = doc['id']
        
        # Index by keywords
        for keyword in doc['keywords']:
            if keyword not in search_index['keywords']:
                search_index['keywords'][keyword] = []
            search_index['keywords'][keyword].append(doc_id)
        
        # Index by drugs
        drugs = doc['drugs'].lower()
        if drugs not in search_index['drugs']:
            search_index['drugs'][drugs] = []
        search_index['drugs'][drugs].append(doc_id)
        
        # Index by source
        source = doc['source']
        if source not in search_index['sources']:
            search_index['sources'][source] = []
        search_index['sources'][source].append(doc_id)
    
    # Save search index
    with open('docs/search_index.json', 'w') as f:
        json.dump(search_index, f, indent=2)
    
    print("✅ Created search index for faster queries")

def main():
    """Main export process."""
    print("🧠 GBM Clinical Database - Browser Export")
    print("=" * 50)
    
    if not Path("vector_db").exists():
        print("❌ Vector database not found!")
        print("Run create_vector_db.py first")
        return
    
    # Export database
    data = export_vector_db_to_json()
    
    if data:
        print("\n✅ Browser export complete!")
        print("📋 Next steps:")
        print("1. Check docs/clinical_database.json")
        print("2. Update GitHub Pages to use client-side processing")
        print("3. Test in browser")
        print("\n🌐 Your GitHub Pages will work without any server!")
    else:
        print("❌ Export failed")

if __name__ == "__main__":
    main()