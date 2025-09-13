#!/usr/bin/env python3
"""
Fine-tune Cross-Encoder Re-ranker for GBM Clinical Domain
Creates training data from existing vector database and fine-tunes cross-encoder
Author: Chetanya Pandey
"""

import chromadb
from chromadb.config import Settings
import json
import pandas as pd
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
import numpy as np
from typing import List, Dict, Tuple
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

class ClinicalRerankerTrainer:
    def __init__(self, db_dir: str = "vector_db"):
        """Initialize the clinical re-ranker trainer."""
        self.db_dir = db_dir
        
        # Connect to existing ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get the collection
        try:
            self.collection = self.chroma_client.get_collection("gbm_clinical_medical_embeddings")
        except:
            self.collection = self.chroma_client.get_collection("gbm_clinical_data")
        
        print(f"✅ Connected to database with {self.collection.count()} documents")
        
        # Clinical query templates for synthetic data generation
        self.query_templates = {
            'dosing': [
                "What is the standard dosing for {drug}?",
                "How do I dose {drug} for {population}?",
                "{drug} dosing protocol for GBM",
                "Recommended {drug} dose and schedule",
                "{drug} mg/m² dosing guidelines"
            ],
            'toxicity': [
                "What are the side effects of {drug}?",
                "{drug} adverse effects and monitoring",
                "How to monitor {drug} toxicity?",
                "{drug} safety profile and warnings",
                "Laboratory monitoring for {drug}"
            ],
            'contraindications': [
                "When should {drug} be avoided?",
                "{drug} contraindications and precautions",
                "Who cannot receive {drug}?",
                "{drug} warnings and contraindications"
            ],
            'administration': [
                "How is {drug} administered?",
                "{drug} administration guidelines",
                "Infusion protocol for {drug}",
                "{drug} preparation and administration"
            ],
            'interactions': [
                "{drug} drug interactions",
                "What drugs interact with {drug}?",
                "Concurrent medications with {drug}",
                "{drug} interaction warnings"
            ]
        }
        
        self.drugs = ['temozolomide', 'bevacizumab', 'TMZ', 'Avastin']
        self.populations = ['newly diagnosed GBM', 'recurrent GBM', 'elderly patients', 'poor performance status']
    
    def generate_clinical_queries(self, n_queries: int = 200) -> List[str]:
        """Generate diverse clinical queries for training."""
        queries = []
        
        # Generate from templates
        for category, templates in self.query_templates.items():
            for template in templates:
                for drug in self.drugs:
                    if '{population}' in template:
                        for pop in self.populations:
                            query = template.format(drug=drug, population=pop)
                            queries.append(query)
                    else:
                        query = template.format(drug=drug)
                        queries.append(query)
        
        # Add domain-specific clinical queries
        clinical_queries = [
            "Standard Stupp protocol dosing",
            "Concurrent chemoradiation dosing",
            "Maintenance temozolomide cycles",
            "Bevacizumab 10 mg/kg dosing",
            "CBC monitoring schedule",
            "Platelet count thresholds",
            "Neutropenia management",
            "Dose reduction criteria",
            "Treatment discontinuation guidelines",
            "MGMT methylation status impact",
            "Performance status considerations",
            "Age-related dose modifications",
            "Renal impairment dosing",
            "Hepatic impairment considerations",
            "Arterial thrombotic events",
            "Hemorrhage risk factors",
            "Wound healing complications",
            "Proteinuria monitoring",
            "Hypertension management",
            "Extended temozolomide treatment",
            "Pseudoprogression vs progression",
            "Quality of life considerations"
        ]
        
        queries.extend(clinical_queries)
        
        # Shuffle and limit
        random.shuffle(queries)
        return queries[:n_queries]
    
    def create_training_data(self, n_queries: int = 150) -> List[InputExample]:
        """Create training examples from clinical queries and document chunks."""
        print("🔄 Generating clinical training queries...")
        queries = self.generate_clinical_queries(n_queries)
        
        training_examples = []
        
        print("🔄 Creating query-document pairs...")
        for i, query in enumerate(queries):
            if i % 20 == 0:
                print(f"   Processing query {i+1}/{len(queries)}")
            
            # Get relevant and irrelevant documents
            relevant_docs, irrelevant_docs = self._get_labeled_documents(query)
            
            # Create positive examples (relevant)
            for doc, score in relevant_docs:
                training_examples.append(InputExample(
                    texts=[query, doc],
                    label=float(score)  # Use relevance score as label
                ))
            
            # Create negative examples (irrelevant)
            for doc, score in irrelevant_docs:
                training_examples.append(InputExample(
                    texts=[query, doc],
                    label=float(score)  # Low relevance score
                ))
        
        print(f"✅ Created {len(training_examples)} training examples")
        return training_examples
    
    def _get_labeled_documents(self, query: str, n_relevant: int = 3, n_irrelevant: int = 2) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Get relevant and irrelevant documents for a query with labels."""
        # Query the database
        results = self.collection.query(
            query_texts=[query],
            n_results=20,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'][0]:
            return [], []
        
        relevant_docs = []
        irrelevant_docs = []
        
        # Process results
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            relevance_score = 1 - distance
            is_relevant = self._is_document_relevant(query, doc, metadata)
            
            if is_relevant and len(relevant_docs) < n_relevant:
                # High relevance score (0.7-1.0)
                adjusted_score = max(0.7, relevance_score)
                relevant_docs.append((doc, adjusted_score))
            elif not is_relevant and len(irrelevant_docs) < n_irrelevant:
                # Low relevance score (0.0-0.3)
                adjusted_score = min(0.3, relevance_score)
                irrelevant_docs.append((doc, adjusted_score))
        
        return relevant_docs, irrelevant_docs
    
    def _is_document_relevant(self, query: str, doc: str, metadata: Dict) -> bool:
        """Determine if document is relevant to query using heuristics."""
        query_lower = query.lower()
        doc_lower = doc.lower()
        
        # Check drug relevance
        if 'temozolomide' in query_lower or 'tmz' in query_lower:
            drug_relevant = any(term in metadata.get('drugs', '').lower() 
                             for term in ['temozolomide', 'tmz'])
            if not drug_relevant:
                return False
        
        if 'bevacizumab' in query_lower or 'avastin' in query_lower:
            drug_relevant = any(term in metadata.get('drugs', '').lower() 
                             for term in ['bevacizumab', 'avastin'])
            if not drug_relevant:
                return False
        
        # Check topic relevance
        topic_matches = {
            'dosing': ['dose', 'dosing', 'mg/m²', 'protocol', 'administration'],
            'toxicity': ['toxicity', 'side effects', 'adverse', 'monitoring', 'safety'],
            'contraindications': ['contraindication', 'avoid', 'warning', 'precaution'],
            'administration': ['administration', 'infusion', 'give', 'inject'],
            'interactions': ['interaction', 'concurrent', 'concomitant']
        }
        
        query_topics = []
        for topic, keywords in topic_matches.items():
            if any(keyword in query_lower for keyword in keywords):
                query_topics.append(topic)
        
        if query_topics:
            # Check if document matches topic
            clinical_topic = metadata.get('clinical_topic', '').lower()
            doc_matches_topic = (clinical_topic in query_topics or 
                               any(any(keyword in doc_lower for keyword in topic_matches[topic]) 
                                   for topic in query_topics))
            return doc_matches_topic
        
        # General relevance based on keyword overlap
        query_keywords = set(query_lower.split())
        doc_keywords = set(doc_lower.split())
        overlap = len(query_keywords.intersection(doc_keywords))
        
        return overlap >= 2  # At least 2 word overlap
    
    def fine_tune_reranker(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2', 
                          output_path: str = 'fine_tuned_gbm_reranker',
                          epochs: int = 3, batch_size: int = 16) -> CrossEncoder:
        """Fine-tune the cross-encoder for clinical domain."""
        
        print(f"🚀 Starting fine-tuning of {model_name}")
        
        # Create training data
        training_examples = self.create_training_data()
        
        # Split into train/validation
        train_examples, val_examples = train_test_split(
            training_examples, test_size=0.2, random_state=42
        )
        
        print(f"📊 Training examples: {len(train_examples)}")
        print(f"📊 Validation examples: {len(val_examples)}")
        
        # Initialize model
        model = CrossEncoder(model_name)
        
        # Create evaluator
        val_queries = {}
        val_corpus = {}
        val_qrels = {}
        
        for i, example in enumerate(val_examples):
            query_id = f"q{i}"
            doc_id = f"d{i}"
            
            val_queries[query_id] = example.texts[0]
            val_corpus[doc_id] = example.texts[1]
            val_qrels[query_id] = {doc_id: int(example.label > 0.5)}
        
        evaluator = CERerankingEvaluator(val_queries, val_corpus, val_qrels)
        
        # Training parameters
        warmup_steps = int(len(train_examples) * epochs * 0.1)
        
        print(f"🔄 Training for {epochs} epochs...")
        print(f"   Batch size: {batch_size}")
        print(f"   Warmup steps: {warmup_steps}")
        
        # Fine-tune
        model.fit(
            train_dataloader=DataLoader(train_examples, shuffle=True, batch_size=batch_size),
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=len(train_examples) // (batch_size * 2),
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True
        )
        
        print(f"✅ Fine-tuning complete! Model saved to: {output_path}")
        
        return model
    
    def evaluate_model(self, model_path: str, test_queries: List[str] = None) -> Dict:
        """Evaluate the fine-tuned model on clinical queries."""
        if test_queries is None:
            test_queries = [
                "temozolomide standard dosing newly diagnosed GBM",
                "bevacizumab side effects monitoring",
                "TMZ dose modification thrombocytopenia",
                "avastin contraindications arterial thrombosis",
                "concurrent chemoradiation protocol"
            ]
        
        # Load fine-tuned model
        fine_tuned_model = CrossEncoder(model_path)
        
        # Load original model for comparison
        original_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        results = {
            'test_queries': test_queries,
            'comparisons': []
        }
        
        print("🧪 Evaluating model performance...")
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            
            # Get documents
            db_results = self.collection.query(
                query_texts=[query],
                n_results=10,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not db_results['documents'][0]:
                continue
            
            docs = db_results['documents'][0][:5]  # Top 5 docs
            
            # Score with both models
            query_doc_pairs = [[query, doc] for doc in docs]
            
            original_scores = original_model.predict(query_doc_pairs)
            fine_tuned_scores = fine_tuned_model.predict(query_doc_pairs)
            
            comparison = {
                'query': query,
                'documents': docs,
                'original_scores': original_scores.tolist(),
                'fine_tuned_scores': fine_tuned_scores.tolist(),
                'score_differences': (fine_tuned_scores - original_scores).tolist()
            }
            
            results['comparisons'].append(comparison)
            
            # Show top result
            orig_best = np.argmax(original_scores)
            ft_best = np.argmax(fine_tuned_scores)
            
            print(f"   Original best (score {original_scores[orig_best]:.3f}): {docs[orig_best][:100]}...")
            print(f"   Fine-tuned best (score {fine_tuned_scores[ft_best]:.3f}): {docs[ft_best][:100]}...")
            
            if orig_best != ft_best:
                print(f"   🔄 Ranking changed!")
        
        return results
    
    def create_benchmark_dataset(self, output_file: str = 'gbm_clinical_benchmark.json'):
        """Create a benchmark dataset for future evaluations."""
        benchmark_queries = [
            # Dosing queries
            "What is the standard temozolomide dose for newly diagnosed GBM?",
            "How should TMZ be dosed during concurrent chemoradiation?",
            "What is the maintenance temozolomide protocol?",
            "What is the recommended bevacizumab dosing for recurrent GBM?",
            "How do you modify TMZ dose for thrombocytopenia?",
            
            # Safety/monitoring queries
            "What laboratory monitoring is required for temozolomide?",
            "How often should CBC be checked during TMZ treatment?",
            "What are the contraindications for bevacizumab?",
            "How do you monitor for bevacizumab-related arterial thrombosis?",
            "When should temozolomide be permanently discontinued?",
            
            # Administration queries
            "How is temozolomide administered?",
            "What is the bevacizumab infusion protocol?",
            "Should TMZ be given with food?",
            "What premedications are needed for avastin?",
            
            # Clinical context queries
            "Does MGMT methylation status affect temozolomide dosing?",
            "Can elderly patients receive standard TMZ dosing?",
            "What is the maximum duration of temozolomide treatment?",
            "When is bevacizumab used in GBM treatment?"
        ]
        
        benchmark_data = []
        
        print("🔄 Creating benchmark dataset...")
        
        for query in benchmark_queries:
            # Get top documents
            results = self.collection.query(
                query_texts=[query],
                n_results=10,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'][0]:
                continue
            
            # Manual relevance labeling (simplified heuristic)
            labeled_docs = []
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                relevance = self._is_document_relevant(query, doc, metadata)
                labeled_docs.append({
                    'document': doc,
                    'metadata': metadata,
                    'distance': distance,
                    'relevant': relevance
                })
            
            benchmark_data.append({
                'query': query,
                'documents': labeled_docs
            })
        
        # Save benchmark
        with open(output_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"✅ Benchmark dataset saved to {output_file}")
        return benchmark_data

def main():
    """Main function to run fine-tuning."""
    trainer = ClinicalRerankerTrainer()
    
    print("🚀 GBM Clinical Re-ranker Fine-tuning")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Fine-tune re-ranker model")
        print("2. Evaluate existing model")
        print("3. Create benchmark dataset")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            model_name = input("Base model (default: cross-encoder/ms-marco-MiniLM-L-6-v2): ").strip()
            if not model_name:
                model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            
            epochs = input("Epochs (default: 3): ").strip()
            epochs = int(epochs) if epochs else 3
            
            print(f"\n🚀 Starting fine-tuning...")
            model = trainer.fine_tune_reranker(
                model_name=model_name,
                epochs=epochs,
                output_path='fine_tuned_gbm_reranker'
            )
            
        elif choice == '2':
            model_path = input("Model path (default: fine_tuned_gbm_reranker): ").strip()
            if not model_path:
                model_path = 'fine_tuned_gbm_reranker'
            
            results = trainer.evaluate_model(model_path)
            
            # Save results
            with open('evaluation_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("✅ Evaluation results saved to evaluation_results.json")
            
        elif choice == '3':
            trainer.create_benchmark_dataset()
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()