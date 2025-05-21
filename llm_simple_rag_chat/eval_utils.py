import pandas as pd
import mlflow
import os

def configure_mlflow(cache_dir=".cache"):
    mlflow_tracking_path = os.path.join(cache_dir, "mlflow")
    # MLflow expects a URI. For local paths, prefix with 'file://'
    mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_tracking_path)}")
    print(f"MLflow tracking data will be stored in: {mlflow_tracking_path}")
    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"


def evaluate_answer(query, answer, reference_answer=None, verbose=True, weight=1.0, cache_dir=".cache"):
    if not reference_answer:
        return None
        
    # Prepare evaluation data
    eval_data = pd.DataFrame({
        "inputs": [query],
        "ground_truth": [reference_answer],
        "model_answer": [answer],
        "weights": [weight]
    })
    
    # Run evaluation
    with mlflow.start_run() as run:
        evaluator = mlflow.evaluate(
            data=eval_data,
            targets="ground_truth",
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
            print(f"Flesch-Kincaid Grade Level: {metrics.get('flesch_kincaid_grade_level/v1/mean', 0.0):.2f}")
            print(f"ARI Grade Level: {metrics.get('ari_grade_level/v1/mean', 0.0):.2f}")
            print("-" * 50)
        
        return {
            "metrics": metrics,
            "eval_table": eval_table
        }
