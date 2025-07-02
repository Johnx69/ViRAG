#!/usr/bin/env python3
"""
Comprehensive evaluation script for Enhanced Vietnamese RAG Pipeline
Author: Assistant
Date: 2024

This script evaluates the enhanced RAG pipeline using multiple metrics:
- GPT-4 based accuracy scoring
- Traditional NLP metrics (ROUGE, BLEU, METEOR)
- Retrieval metrics (Precision, Recall, F1)
- Context relevance evaluation

Usage:
    python evaluate_rag_pipeline.py

Outputs:
    - results/evaluation_results.json: Detailed results for each question
    - results/metrics_summary.txt: Overall metrics summary
    - results/detailed_metrics.txt: Detailed metrics breakdown
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import traceback
from datetime import datetime

# Import necessary libraries for metrics
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    from sklearn.metrics import precision_score, recall_score, f1_score
    from openai import OpenAI
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Missing required packages. Please install them using:")
    print("pip install rouge-score nltk scikit-learn openai numpy pandas")
    print(f"Error: {e}")
    sys.exit(1)

# Import your RAG pipeline
sys.path.append("/Users/hoanganh692004/Desktop/RAG_project/rag_project")
try:
    from src import EnhancedVietnameseRAGPipeline, setup_logging
    from src.config.settings import settings
except ImportError as e:
    print(f"Error importing RAG pipeline: {e}")
    print(
        "Please make sure the project path is correct and the pipeline is properly installed."
    )
    sys.exit(1)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class RAGEvaluator:
    """Comprehensive RAG pipeline evaluator"""

    def __init__(self, test_data_path: str, results_dir: str = "results"):
        self.test_data_path = Path(test_data_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Setup logging first
        self.setup_logging()

        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.smoothing_function = SmoothingFunction().method1

        # Test NLTK tokenization
        self.test_nltk_resources()

        # Initialize GPT-4 for evaluation
        try:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

        # Initialize RAG pipeline
        self.pipeline = None
        self.init_pipeline()

        # Results storage
        self.detailed_results = []
        self.metrics_summary = {}

    def test_nltk_resources(self):
        """Test if NLTK resources are available"""
        try:
            # Test tokenization
            test_text = "This is a test sentence."
            tokens = nltk.word_tokenize(test_text)
            self.logger.info("NLTK tokenization test passed")
        except Exception as e:
            self.logger.warning(f"NLTK tokenization test failed: {e}")
            self.logger.info("Will use fallback tokenization methods")

    def setup_logging(self):
        """Setup logging configuration"""
        log_file = str(self.results_dir / "evaluation.log")

        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging format
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Configure handlers
        handlers = [logging.StreamHandler(sys.stdout)]
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=handlers,
            force=True,  # Override any existing configuration
        )

        # Reduce verbosity of external libraries
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("weaviate").setLevel(logging.WARNING)

        self.logger = logging.getLogger(__name__)

    def init_pipeline(self):
        """Initialize the enhanced RAG pipeline"""
        try:
            self.logger.info("Initializing Enhanced Vietnamese RAG Pipeline...")
            self.pipeline = EnhancedVietnameseRAGPipeline()
            self.logger.info("Pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise

    def load_test_data(self) -> List[Dict[str, str]]:
        """Load test data from JSONL file"""
        test_data = []
        try:
            with open(self.test_data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        test_data.append(data)
            self.logger.info(f"Loaded {len(test_data)} test questions")
            return test_data
        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            raise

    def get_pipeline_answer(self, question: str) -> Dict[str, Any]:
        """Get answer from the enhanced RAG pipeline"""
        try:
            # Use specific parameters for evaluation
            result = self.pipeline.query(
                question=question,
                rewrite_strategies=["simple", "decompose", "hyde"],
                max_web_searches=1,  # Limit web searches for consistency
                max_return_docs=1,  # k=1 as requested
            )
            return result
        except Exception as e:
            self.logger.error(f"Error getting pipeline answer for '{question}': {e}")
            return {
                "answer": "Error occurred during pipeline execution",
                "source": "error",
                "error": str(e),
                "retrieved_docs": [],
                "confidence_score": 0.0,
            }

    def evaluate_with_gpt4(
        self, question: str, generated_answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Evaluate answer accuracy using GPT-4"""
        prompt = f"""
        Bạn là một chuyên gia đánh giá chất lượng câu trả lời. Hãy đánh giá câu trả lời được tạo ra so với câu trả lời chuẩn.

        Câu hỏi: {question}
        
        Câu trả lời chuẩn: {ground_truth}
        
        Câu trả lời được tạo: {generated_answer}
        
        Hãy đánh giá theo các tiêu chí sau (thang điểm 1-5):
        1. Tính chính xác về mặt sự kiện (Factual Accuracy)
        2. Tính đầy đủ của thông tin (Completeness)
        3. Tính liên quan đến câu hỏi (Relevance)
        4. Chất lượng ngôn ngữ (Language Quality)
        
        Trả về kết quả theo định dạng JSON:
        {{
            "factual_accuracy": <điểm 1-5>,
            "completeness": <điểm 1-5>,
            "relevance": <điểm 1-5>,
            "language_quality": <điểm 1-5>,
            "overall_score": <điểm trung bình>,
            "explanation": "<giải thích ngắn gọn>"
        }}
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content

            # Extract JSON from response
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_text = result_text[start:end]
                result = json.loads(json_text)
                # Convert numpy types to native Python types
                return convert_numpy_types(result)
            else:
                raise ValueError("No valid JSON found in GPT-4 response")

        except Exception as e:
            self.logger.error(f"Error in GPT-4 evaluation: {e}")
            return {
                "factual_accuracy": 0,
                "completeness": 0,
                "relevance": 0,
                "language_quality": 0,
                "overall_score": 0,
                "explanation": f"Error in evaluation: {str(e)}",
            }

    def calculate_rouge_scores(
        self, generated: str, reference: str
    ) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, generated)
            # Convert to native Python float types
            return {
                "rouge1_f": float(scores["rouge1"].fmeasure),
                "rouge1_p": float(scores["rouge1"].precision),
                "rouge1_r": float(scores["rouge1"].recall),
                "rouge2_f": float(scores["rouge2"].fmeasure),
                "rouge2_p": float(scores["rouge2"].precision),
                "rouge2_r": float(scores["rouge2"].recall),
                "rougeL_f": float(scores["rougeL"].fmeasure),
                "rougeL_p": float(scores["rougeL"].precision),
                "rougeL_r": float(scores["rougeL"].recall),
            }
        except Exception as e:
            self.logger.error(f"Error calculating ROUGE scores: {e}")
            return {
                key: 0.0
                for key in [
                    "rouge1_f",
                    "rouge1_p",
                    "rouge1_r",
                    "rouge2_f",
                    "rouge2_p",
                    "rouge2_r",
                    "rougeL_f",
                    "rougeL_p",
                    "rougeL_r",
                ]
            }

    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU score with fallback tokenization"""
        try:
            # Try NLTK tokenization first
            try:
                reference_tokens = [nltk.word_tokenize(reference.lower())]
                generated_tokens = nltk.word_tokenize(generated.lower())
            except Exception as e:
                # Fallback to simple whitespace tokenization
                self.logger.warning(
                    f"NLTK tokenization failed, using simple tokenization: {e}"
                )
                reference_tokens = [reference.lower().split()]
                generated_tokens = generated.lower().split()

            score = sentence_bleu(
                reference_tokens,
                generated_tokens,
                smoothing_function=self.smoothing_function,
            )
            # Ensure native Python float
            return float(score)
        except Exception as e:
            self.logger.error(f"Error calculating BLEU score: {e}")
            return 0.0

    def calculate_meteor_score(self, generated: str, reference: str) -> float:
        """Calculate METEOR score with fallback tokenization"""
        try:
            # Try NLTK tokenization first
            try:
                reference_tokens = nltk.word_tokenize(reference.lower())
                generated_tokens = nltk.word_tokenize(generated.lower())
            except Exception as e:
                # Fallback to simple whitespace tokenization
                self.logger.warning(
                    f"NLTK tokenization failed for METEOR, using simple tokenization: {e}"
                )
                reference_tokens = reference.lower().split()
                generated_tokens = generated.lower().split()

            score = meteor_score([reference_tokens], generated_tokens)
            # Ensure native Python float
            return float(score)
        except Exception as e:
            self.logger.error(f"Error calculating METEOR score: {e}")
            return 0.0

    def evaluate_retrieval_quality(
        self, retrieved_docs: List[Dict], ground_truth_context: str
    ) -> Dict[str, float]:
        """Evaluate retrieval quality with k=1"""
        try:
            if not retrieved_docs:
                return {
                    "retrieval_precision": 0.0,
                    "retrieval_recall": 0.0,
                    "retrieval_f1": 0.0,
                    "context_relevance": 0.0,
                }

            # For k=1, we only have one retrieved document
            retrieved_doc = retrieved_docs[0]
            retrieved_content = retrieved_doc.get("content", "")

            # Simple overlap-based evaluation
            ground_truth_words = set(ground_truth_context.lower().split())
            retrieved_words = set(retrieved_content.lower().split())

            # Calculate overlap
            intersection = ground_truth_words.intersection(retrieved_words)

            if len(retrieved_words) == 0:
                precision = 0.0
            else:
                precision = len(intersection) / len(retrieved_words)

            if len(ground_truth_words) == 0:
                recall = 0.0
            else:
                recall = len(intersection) / len(ground_truth_words)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            # Context relevance based on content overlap
            context_relevance = min(
                1.0, len(intersection) / max(1, len(ground_truth_words) * 0.3)
            )

            # Ensure all values are native Python floats
            return {
                "retrieval_precision": float(precision),
                "retrieval_recall": float(recall),
                "retrieval_f1": float(f1),
                "context_relevance": float(context_relevance),
            }

        except Exception as e:
            self.logger.error(f"Error evaluating retrieval quality: {e}")
            return {
                "retrieval_precision": 0.0,
                "retrieval_recall": 0.0,
                "retrieval_f1": 0.0,
                "context_relevance": 0.0,
            }

    def evaluate_single_question(
        self, test_item: Dict[str, str], question_id: int
    ) -> Dict[str, Any]:
        """Evaluate a single question comprehensively"""
        question = test_item["question"]
        ground_truth_answer = test_item["answer"]
        ground_truth_context = test_item["context"]

        self.logger.info(f"Evaluating question {question_id + 1}: {question[:50]}...")

        # Get pipeline answer
        pipeline_result = self.get_pipeline_answer(question)
        generated_answer = pipeline_result.get("answer", "")

        # GPT-4 evaluation
        gpt4_scores = self.evaluate_with_gpt4(
            question, generated_answer, ground_truth_answer
        )

        # Traditional NLP metrics
        rouge_scores = self.calculate_rouge_scores(
            generated_answer, ground_truth_answer
        )
        bleu_score = self.calculate_bleu_score(generated_answer, ground_truth_answer)
        meteor_score_val = self.calculate_meteor_score(
            generated_answer, ground_truth_answer
        )

        # Retrieval evaluation
        retrieved_docs = pipeline_result.get("retrieved_docs", [])
        retrieval_scores = self.evaluate_retrieval_quality(
            retrieved_docs, ground_truth_context
        )

        # Compile results - ensure all values are JSON serializable
        result = {
            "question_id": question_id + 1,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "ground_truth_context": ground_truth_context,
            "generated_answer": generated_answer,
            "pipeline_info": {
                "source": pipeline_result.get("source", ""),
                "confidence_score": float(pipeline_result.get("confidence_score", 0.0)),
                "num_sources": int(pipeline_result.get("num_sources", 0)),
                "process_log": pipeline_result.get("process_log", []),
                "rewrite_info": pipeline_result.get("rewrite_info", {}),
                "retrieved_docs": retrieved_docs,
            },
            "gpt4_evaluation": gpt4_scores,
            "traditional_metrics": {
                "bleu_score": bleu_score,
                "meteor_score": meteor_score_val,
                **rouge_scores,
            },
            "retrieval_metrics": retrieval_scores,
            "timestamp": datetime.now().isoformat(),
        }

        # Convert any remaining numpy types
        result = convert_numpy_types(result)
        return result

    def run_evaluation(self):
        """Run complete evaluation"""
        self.logger.info("Starting comprehensive RAG evaluation...")

        # Load test data
        test_data = self.load_test_data()

        # Evaluate each question
        for i, test_item in enumerate(test_data):
            try:
                result = self.evaluate_single_question(test_item, i)
                self.detailed_results.append(result)

                # Log progress
                if (i + 1) % 5 == 0 or (i + 1) == len(test_data):
                    self.logger.info(f"Completed {i + 1}/{len(test_data)} questions")

            except Exception as e:
                self.logger.error(f"Error evaluating question {i + 1}: {e}")
                traceback.print_exc()

        # Calculate summary metrics
        self.calculate_summary_metrics()

        # Save results
        self.save_results()

        self.logger.info("Evaluation completed successfully!")

    def calculate_summary_metrics(self):
        """Calculate overall summary metrics"""
        if not self.detailed_results:
            return

        metrics = {
            "total_questions": len(self.detailed_results),
            "evaluation_timestamp": datetime.now().isoformat(),
        }

        # GPT-4 metrics
        gpt4_metrics = [
            "factual_accuracy",
            "completeness",
            "relevance",
            "language_quality",
            "overall_score",
        ]
        for metric in gpt4_metrics:
            scores = [
                r["gpt4_evaluation"][metric]
                for r in self.detailed_results
                if metric in r["gpt4_evaluation"]
            ]
            if scores:
                # Ensure native Python float types
                metrics[f"avg_gpt4_{metric}"] = float(np.mean(scores))
                metrics[f"std_gpt4_{metric}"] = float(np.std(scores))

        # Traditional NLP metrics
        nlp_metrics = ["bleu_score", "meteor_score", "rouge1_f", "rouge2_f", "rougeL_f"]
        for metric in nlp_metrics:
            scores = [
                r["traditional_metrics"][metric]
                for r in self.detailed_results
                if metric in r["traditional_metrics"]
            ]
            if scores:
                # Ensure native Python float types
                metrics[f"avg_{metric}"] = float(np.mean(scores))
                metrics[f"std_{metric}"] = float(np.std(scores))

        # Retrieval metrics
        retrieval_metrics = [
            "retrieval_precision",
            "retrieval_recall",
            "retrieval_f1",
            "context_relevance",
        ]
        for metric in retrieval_metrics:
            scores = [
                r["retrieval_metrics"][metric]
                for r in self.detailed_results
                if metric in r["retrieval_metrics"]
            ]
            if scores:
                # Ensure native Python float types
                metrics[f"avg_{metric}"] = float(np.mean(scores))
                metrics[f"std_{metric}"] = float(np.std(scores))

        # Source distribution
        sources = [r["pipeline_info"]["source"] for r in self.detailed_results]
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        metrics["source_distribution"] = source_counts

        # Convert any remaining numpy types
        self.metrics_summary = convert_numpy_types(metrics)

    def save_results(self):
        """Save all results to files"""
        # Convert detailed results to ensure JSON serialization
        json_safe_results = convert_numpy_types(self.detailed_results)

        # Save detailed results as JSON
        detailed_file = self.results_dir / "evaluation_results.json"
        with open(detailed_file, "w", encoding="utf-8") as f:
            json.dump(json_safe_results, f, ensure_ascii=False, indent=2)

        # Save summary metrics as TXT
        summary_file = self.results_dir / "metrics_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("=== RAG PIPELINE EVALUATION SUMMARY ===\n\n")
            f.write(
                f"Evaluation Date: {self.metrics_summary.get('evaluation_timestamp', 'N/A')}\n"
            )
            f.write(
                f"Total Questions: {self.metrics_summary.get('total_questions', 0)}\n\n"
            )

            f.write("=== GPT-4 EVALUATION METRICS ===\n")
            gpt4_metrics = [
                "factual_accuracy",
                "completeness",
                "relevance",
                "language_quality",
                "overall_score",
            ]
            for metric in gpt4_metrics:
                avg_key = f"avg_gpt4_{metric}"
                std_key = f"std_gpt4_{metric}"
                if avg_key in self.metrics_summary:
                    f.write(
                        f"{metric.replace('_', ' ').title()}: {self.metrics_summary[avg_key]:.3f} ± {self.metrics_summary.get(std_key, 0):.3f}\n"
                    )

            f.write("\n=== TRADITIONAL NLP METRICS ===\n")
            nlp_metrics = [
                "bleu_score",
                "meteor_score",
                "rouge1_f",
                "rouge2_f",
                "rougeL_f",
            ]
            for metric in nlp_metrics:
                avg_key = f"avg_{metric}"
                std_key = f"std_{metric}"
                if avg_key in self.metrics_summary:
                    f.write(
                        f"{metric.replace('_', ' ').upper()}: {self.metrics_summary[avg_key]:.3f} ± {self.metrics_summary.get(std_key, 0):.3f}\n"
                    )

            f.write("\n=== RETRIEVAL METRICS (k=1) ===\n")
            retrieval_metrics = [
                "retrieval_precision",
                "retrieval_recall",
                "retrieval_f1",
                "context_relevance",
            ]
            for metric in retrieval_metrics:
                avg_key = f"avg_{metric}"
                std_key = f"std_{metric}"
                if avg_key in self.metrics_summary:
                    f.write(
                        f"{metric.replace('_', ' ').title()}: {self.metrics_summary[avg_key]:.3f} ± {self.metrics_summary.get(std_key, 0):.3f}\n"
                    )

            f.write("\n=== SOURCE DISTRIBUTION ===\n")
            source_dist = self.metrics_summary.get("source_distribution", {})
            for source, count in source_dist.items():
                percentage = (
                    count / self.metrics_summary.get("total_questions", 1)
                ) * 100
                f.write(f"{source}: {count} ({percentage:.1f}%)\n")

        # Save detailed metrics breakdown
        detailed_metrics_file = self.results_dir / "detailed_metrics.txt"
        with open(detailed_metrics_file, "w", encoding="utf-8") as f:
            f.write("=== DETAILED METRICS BREAKDOWN ===\n\n")

            for i, result in enumerate(self.detailed_results, 1):
                f.write(f"Question {i}: {result['question']}\n")
                f.write(f"Generated Answer: {result['generated_answer'][:100]}...\n")
                f.write(f"Ground Truth: {result['ground_truth_answer']}\n")

                f.write("GPT-4 Scores: ")
                gpt4 = result["gpt4_evaluation"]
                f.write(f"Overall={gpt4.get('overall_score', 0):.2f}, ")
                f.write(f"Accuracy={gpt4.get('factual_accuracy', 0):.2f}, ")
                f.write(f"Relevance={gpt4.get('relevance', 0):.2f}\n")

                f.write("Traditional Metrics: ")
                trad = result["traditional_metrics"]
                f.write(f"BLEU={trad.get('bleu_score', 0):.3f}, ")
                f.write(f"ROUGE-L={trad.get('rougeL_f', 0):.3f}\n")

                f.write("Retrieval Metrics: ")
                retr = result["retrieval_metrics"]
                f.write(f"Precision={retr.get('retrieval_precision', 0):.3f}, ")
                f.write(f"Recall={retr.get('retrieval_recall', 0):.3f}, ")
                f.write(f"F1={retr.get('retrieval_f1', 0):.3f}\n")

                f.write(f"Source: {result['pipeline_info']['source']}\n")
                f.write("-" * 80 + "\n\n")

        self.logger.info(f"Results saved to {self.results_dir}")
        print(f"\n=== EVALUATION COMPLETE ===")
        print(f"Results saved to: {self.results_dir}")
        print(f"- Detailed results: {detailed_file}")
        print(f"- Summary metrics: {summary_file}")
        print(f"- Detailed breakdown: {detailed_metrics_file}")


def ensure_nltk_data():
    """Ensure NLTK data is downloaded"""
    print("Checking and downloading required NLTK data...")

    # Download required NLTK data
    resources = ["punkt", "punkt_tab", "wordnet", "omw-1.4"]
    success_count = 0

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"✓ Downloaded {resource}")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to download {resource}: {e}")

    # Test tokenization
    try:
        test_tokens = nltk.word_tokenize("Test sentence.")
        print("✓ NLTK tokenization working")
        return True
    except Exception as e:
        print(f"✗ NLTK tokenization test failed: {e}")

        # If downloads failed, suggest manual download
        if success_count < len(resources):
            print("\n" + "=" * 50)
            print("NLTK DATA DOWNLOAD ISSUE DETECTED")
            print("=" * 50)
            print("If you continue to see NLTK errors, try running this manually:")
            print(
                "python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')\""
            )
            print("=" * 50 + "\n")

        print("Will use fallback tokenization methods")
        return False


def main():
    """Main execution function"""
    print("=== RAG PIPELINE EVALUATION ===\n")

    # Ensure NLTK data is available
    ensure_nltk_data()

    # Configuration
    test_data_path = "/Users/hoanganh692004/Desktop/RAG_project/rag_project/data/test/test_data_wiki.jsonl"
    results_dir = "results"

    # Validate test data path
    if not Path(test_data_path).exists():
        print(f"Error: Test data file not found at {test_data_path}")
        print("Please check the path and try again.")
        sys.exit(1)

    # Create evaluator and run evaluation
    try:
        print(f"\nInitializing evaluator...")
        evaluator = RAGEvaluator(test_data_path, results_dir)
        print("Starting evaluation...")
        evaluator.run_evaluation()
    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
