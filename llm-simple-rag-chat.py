import sys
import os
import time
import argparse
import logging
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
from datetime import datetime

from llm_simple_rag_chat.document_utils import (
    load_and_cache_chunks
)

from llm_simple_rag_chat.rag_utils import (
    build_rag_system
)

from llm_simple_rag_chat.genai_utils import (
    setup_genai_environment,
    validate_model,
    create_llm
)

from llm_simple_rag_chat.eval_utils import (
    analyze_evaluation_results,
    evaluate_answer,
    configure_mlflow
)

# Suppress TensorFlow and XLA logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

exectime_internal = 0.0
exectime_external = 0.0
time_start = time.time()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple chat application with RAG and answers evaluations")

    g = parser.add_argument_group('Generic options')
    g.add_argument('-v', '--verbose', action='count', help='enable verbose mode (use -vv for max verbosity)')
    g.add_argument('-l', '--logfile', action='store', help='log filename')

    g = parser.add_argument_group('Model options')
    g.add_argument('--reasoning-model', 
        default="models/gemini-2.0-flash",
        help="Gemini model for reasoning/chat"
    )
    g.add_argument('--embedding-model', 
        default="models/embedding-001",
        help="Gemini model for embeddings"
    )
    g.add_argument('--list-models',
        action="store_true",
        help="List available Google models and exit (useful for validating API token and selecting models)"
    )

    g = parser.add_argument_group('Document options')
    g.add_argument('--documents-folder',
        default="./documents",
        help="Path to the documents folder"
    )

    g = parser.add_argument_group('Evaluation options')
    g.add_argument('--analyze-results',
        action="store_true",
        help="Analyze existing evaluation results and print summary statistics"
    )
    g.add_argument('--results-folder',
        default=".results",
        help="Path to the folder containing evaluation results"
    )

    g = parser.add_argument_group('Mode options')
    g.add_argument('--mode',
        choices=["interactive", "auto"],
        default="interactive",
        help="Mode of operation: 'interactive' for manual chat or 'auto' for automated questions"
    )
    g.add_argument('--questions-file',
        default="questions.json",
        help="Path to the questions JSON file"
    )
    g.add_argument('--cache-dir',
        default=".cache",
        help="Directory to store cached artifacts and data"
    )

    return parser.parse_args()

# Auto mode handler
def process_auto_mode(qa_chain, questions_file, cache_dir, args):  
    # Load questions from JSON file
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    # Initialize result file
    output_path, output_data = initialize_result_file(args, questions_file)
    output_data["categories"] = {}
    
    # Process each category
    for category, category_data in questions_data['categories'].items():
        print(f"\nProcessing category: {category}")
        output_data['categories'][category] = {"questions": []}
        
        # Process each question in the category
        for question_data in category_data['questions']:
            print(f"\nProcessing question: {question_data['question']}")
            
            # Get answer from AI
            response = qa_chain.invoke({"query": question_data['question']})
            answer = response['result']
            
            # Run evaluation with cache directory
            eval_results = evaluate_answer(
                question_data['question'],
                answer,
                question_data['reference_answer'],
                verbose=False,  # Don't print results in auto mode
                weight=question_data['weight'],
                cache_dir=cache_dir
            )
            
            # Create output question data
            output_question = {
                "question": question_data['question'],
                "reference_answer": question_data['reference_answer'],
                "weight": question_data['weight'],
                "model_answer": answer,
                "eval_results": {
                    "metrics": eval_results['metrics'] if eval_results else {},
                    "eval_table": eval_results['eval_table'].to_dict() if eval_results else {}
                }
            }
            
            # Add to output data
            output_data['categories'][category]['questions'].append(output_question)
            
            # Print summary metrics
            if eval_results:
                metrics = eval_results['metrics']
                print(f"Evaluation metrics: {metrics}")
            
        # Save evaluation results
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nEvaluation results saved to: {output_path}")
        print("All questions processed successfully!")
    return output_path, output_data

def initialize_result_file(args, source_file=None):
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path('.results')
    results_dir.mkdir(exist_ok=True)
    
    # Generate output filename based on mode
    if args.mode == "auto":
        output_filename = f"{timestamp}_{Path(source_file).stem}_evaluation_results_{args.embedding_model.replace('/', '_')}.json"
    else:
        output_filename = f"{timestamp}_interactive_session_{args.embedding_model.replace('/', '_')}.json"
    
    # Create output path
    output_path = results_dir / output_filename
    
    # Initialize output data structure
    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "model": args.embedding_model,
            "mode": args.mode,
            "source_file": source_file
        }
    }
    
    return output_path, output_data


def analyze_evaluation_results(results_folder):
    """
    Analyze evaluation results from the results folder and print statistics per file.
    """
    import json
    from pathlib import Path
    import pandas as pd
    from collections import defaultdict

    results_dir = Path(results_folder)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return

    # Process each result file
    for result_file in sorted(results_dir.glob("*.json")):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Get file metadata
            timestamp = data["metadata"]["timestamp"]
            model = data["metadata"]["model"]
            mode = data["metadata"]["mode"]
            source_file = data["metadata"].get("source_file", "N/A")

            print(f"\n=== Results for {result_file.name} ===")
            print(f"Timestamp: {timestamp}")
            print(f"Model: {model}")
            print(f"Mode: {mode}")
            print(f"Source file: {source_file}")

            # Initialize statistics for this file
            file_stats = defaultdict(lambda: {"total_weight": 0, "total_score": 0, "questions": []})
            file_scores = []

            # Process auto mode results
            if "categories" in data:
                for category, category_data in data["categories"].items():
                    for question in category_data["questions"]:
                        # Calculate normalized relevance score (0-1)
                        metrics = question["eval_results"]["metrics"]
                        relevance_score = metrics.get("exact_match/v1", 0.0)
                        
                        # Add to category statistics
                        file_stats[category]["total_weight"] += question["weight"]
                        file_stats[category]["total_score"] += relevance_score * question["weight"]
                        file_stats[category]["questions"].append({
                            "question": question["question"],
                            "score": relevance_score,
                            "weight": question["weight"]
                        })
                        
                        file_scores.append(relevance_score)

            # Process interactive mode results
            elif "questions" in data:
                for question in data["questions"]:
                    metrics = question["eval_results"]["metrics"]
                    relevance_score = metrics.get("exact_match/v1", 0.0)
                    
                    file_stats["Interactive"]["total_weight"] += 1.0  # Interactive mode uses equal weights
                    file_stats["Interactive"]["total_score"] += relevance_score
                    file_stats["Interactive"]["questions"].append({
                        "question": question["question"],
                        "score": relevance_score,
                        "weight": 1.0
                    })
                    
                    file_scores.append(relevance_score)

            # Print file-level statistics
            print("\n=== File Statistics ===")
            print(f"Total questions: {sum(len(v["questions"]) for v in file_stats.values())}")
            print(f"Average score: {sum(file_scores)/len(file_scores):.3f}")

            # Create DataFrame for file-level statistics
            file_df = pd.DataFrame({
                "Category": [],
                "Questions": [],
                "Average Score": [],
                "Weighted Score": []
            })

            # Print per-category statistics for this file
            for category, stats in file_stats.items():
                if stats["total_weight"] == 0:
                    continue

                avg_score = stats["total_score"] / len(stats["questions"])
                weighted_score = stats["total_score"] / stats["total_weight"]

                # Add to DataFrame
                file_df = pd.concat([
                    file_df,
                    pd.DataFrame({
                        "Category": [category],
                        "Questions": [len(stats["questions"])],
                        "Average Score": [avg_score],
                        "Weighted Score": [weighted_score]
                    })
                ], ignore_index=True)

                # Print detailed category information
                print(f"\nCategory: {category}")
                print(f"Questions evaluated: {len(stats['questions'])}")
                print(f"Average score: {avg_score:.3f}")
                print(f"Weighted score: {weighted_score:.3f}")

            # Print file-level statistics table
            print("\n=== Detailed Statistics by Category ===")
            print(file_df.to_string(index=False))

            # Print score distribution for this file
            if file_scores:
                score_series = pd.Series(file_scores)
                
                # Create bins manually
                bins = [0, 0.25, 0.5, 0.75, 1.0]
                labels = ['0-25%', '25-50%', '50-75%', '75-100%']
                
                # Count scores in each bin
                binned_scores = pd.cut(score_series, bins=bins, labels=labels)
                bin_counts = binned_scores.value_counts()
                
                # Calculate percentages
                total_scores = len(file_scores)
                bin_percentages = (bin_counts / total_scores) * 100
                
                print("\n=== Score Distribution ===")
                print(bin_percentages.to_string())

        except Exception as e:
            print(f"Error processing {result_file}: {str(e)}")
            continue


def run_interactive_mode(qa_chain, cache_dir, args):
    print("\nReady to answer questions! Type 'exit' to quit.")
    
    # Initialize interactive session data
    output_path, session_data = initialize_result_file(args)
    session_data["questions"] = []
    
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'exit':
            # Save session data before exiting
            with open(output_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            print(f"\nInteractive session saved to: {output_path}")
            break
        
        response = qa_chain.invoke({"query": query})
        answer = response['result']
        print(f"\nAI's Answer: {answer}")
        print("\nSources:")
        print("    " + "-" * 50)  # Add a separator line with indentation
        for i, doc in enumerate(response['source_documents']):
            print(f"--- Document {i+1} ---")
            # Extract document path and try to identify section
            doc_path = doc.metadata.get('source', 'N/A')
            
            # Try to extract section information
            section = "Unknown section"
            content_lines = doc.page_content.split('\n')
            
            # Look for potential section headers in the first few lines
            for i, line in enumerate(content_lines[:5]):
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Check for markdown headers
                if line.startswith('#'):
                    section = line.strip('# \t')
                    break
                    
            print(f"    Path: {doc_path}")
            print(f"    Section: {section}\n")

        # Get reference answer for evaluation
        reference_answer = input("\nPlease provide the reference answer for evaluation (or press Enter to skip): ")
        if reference_answer:
            eval_results = evaluate_answer(query, answer, reference_answer, cache_dir=cache_dir)
            
            # Add question data to session
            session_data['questions'].append({
                "question": query,
                "reference_answer": reference_answer,
                "model_answer": answer,
                "eval_results": {
                    "metrics": eval_results['metrics'] if eval_results else {},
                    "eval_table": eval_results['eval_table'].to_dict() if eval_results else {}
                }
            })
            
            # Print evaluation metrics
            if eval_results:
                metrics = eval_results['metrics']
                print(f"\nEvaluation metrics: {metrics}")


def main():
    args = parse_arguments()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.logfile) if args.logfile else logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # List models if requested
    if args.list_models:
        setup_genai_environment()
        validate_model(args.embedding_model)
        return

    # Analyze results if requested
    if args.analyze_results:
        from llm_simple_rag_chat.eval_utils import analyze_evaluation_results
        analyze_evaluation_results(args.results_folder)
        return

    # Ensure cache directory exists
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created cache directory at: {cache_dir}")

    # Setup Google Generative AI environment
    setup_genai_environment()
    api_key, llm = create_llm(args)

    # Configure MLflow
    configure_mlflow(args.cache_dir)

    # Load and cache document chunks
    chunks, changed = load_and_cache_chunks(args.documents_folder, args.cache_dir)
    print(f"\nDocument chunks loaded. Changes detected: {changed}")
    
    # Create LLM and build RAG system

    qa_chain = build_rag_system(args.embedding_model, api_key, chunks, llm, args.cache_dir)

    # Run in selected mode
    if args.mode == "interactive":
        run_interactive_mode(qa_chain, args.cache_dir, args)
    else:
        process_auto_mode(qa_chain, args.questions_file, args.cache_dir, args)

    # Log execution time
    time_end = time.time()
    logging.info(f"Execution time: {time_end - time_start:.2f} seconds")

if __name__ == "__main__":
    main()