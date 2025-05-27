import pandas as pd
import mlflow
import os
import requests
from mlflow.metrics.genai import answer_similarity, answer_correctness, answer_relevance, relevance, faithfulness

def configure_mlflow(cache_dir=".cache", llm_as_a_judge=False, ollama_address="http://localhost:11434"):
    mlflow_tracking_path = os.path.join(cache_dir, "mlflow")
    # MLflow expects a URI. For local paths, prefix with 'file://'
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_tracking_path)}")
    print(f"MLflow tracking data will be stored in: {mlflow_tracking_path}")
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

    # Configure MLflow environment variables for OpenAI adapter
    os.environ["OPENAI_API_BASE"] = f"{ollama_address}/v1"
    os.environ["OPENAI_API_KEY"] = "qwen"  # Required by the OpenAI client library

    # Check Ollama accessibility if llm_as_a_judge is True
    if llm_as_a_judge:
        try:
            response = requests.get(f"{ollama_address}/api/tags")
            if response.status_code != 200:
                raise Exception("Ollama API is not accessible")
            print("Ollama is accessible and ready for evaluation")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama. Please ensure Ollama is running on {ollama_address}")

def evaluate_answer(query, answer, reference_answer=None, source_documents=None, verbose=True, weight=1.0, cache_dir=".cache", llm_as_a_judge=False, model_name="qwen3:8b"):
    if not reference_answer:
        return None
        
    # Prepare evaluation data
    eval_data = pd.DataFrame({
        "inputs": [query],
        "ground_truth": [reference_answer],
        "model_answer": [answer],
        "weights": [weight]
    })

    # Add source documents if provided
    if source_documents:
        # Combine all document contents into a single context string
        context = "\n".join([doc.page_content for doc in source_documents])
        eval_data["context"] = [context]

    def create_metric_config(metric_func):
        """Helper function to create metric configuration with common parameters."""
        return metric_func(
            model=f"openai:/{model_name}",
            parameters={
                "max_tokens": 2048,
                "temperature": 0.0,
                "top_p": 1,
            },
            max_workers=1,
        )

    # Prepare metrics list based on llm_as_a_judge flag
    extra_metrics = []
    if llm_as_a_judge:
        extra_metrics.extend([
            create_metric_config(answer_similarity),
            create_metric_config(answer_correctness),
            create_metric_config(answer_relevance),
        ])
        if source_documents:
            extra_metrics.extend([
                create_metric_config(relevance),
                create_metric_config(faithfulness),
            ])

    # Run evaluation
    with mlflow.start_run() as run:
        evaluator = mlflow.evaluate(
            data=eval_data,
            targets="ground_truth",
            extra_metrics=extra_metrics,
            predictions="model_answer",
            model_type="question-answering"
        )
        
        # Get evaluation results
        eval_table = evaluator.tables["eval_results_table"]
        metrics = evaluator.metrics
        
        if verbose:
            print("\nEvaluation Results:")
            print("-" * 50)
            print(f"Exact Match Score: {metrics.get('exact_match/v1', 0.0)}")
            if llm_as_a_judge:
                print(f"Answer Similarity Score: {metrics.get('answer_similarity/v1/mean', 0.0)}")
                print(f"Answer Correctness Score: {metrics.get('answer_correctness/v1/mean', 0.0)}")
                print(f"Answer Relevance Score: {metrics.get('answer_relevance/v1/mean', 0.0)}")
                if source_documents:
                    print(f"Relevance Score: {metrics.get('relevance/v1/mean', 0.0)}")
                    print(f"Faithfulness Score: {metrics.get('faithfulness/v1/mean', 0.0)}")
            print(f"Flesch-Kincaid Grade Level: {metrics.get('flesch_kincaid_grade_level/v1/mean', 0.0):.2f}")
            print(f"ARI Grade Level: {metrics.get('ari_grade_level/v1/mean', 0.0):.2f}")
            print("-" * 50)
        
        return {
            "metrics": metrics,
            "eval_table": eval_table
        }
