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
    evaluate_answers,
    configure_mlflow
)

from llm_simple_rag_chat.results_analysis import (
    analyze_evaluation_results
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
    g.add_argument('--temperature', type=float, default=0.1,
        help="Model temperature for controlling randomness in responses (0.0 = deterministic, 2.0 = more random)"
    )
    g.add_argument('--n-tokens', type=int, default=1024,
        help="Maximum number of tokens for model responses"
    )
    g.add_argument('--top-p', type=float, default=0.95,
        help="Top-p sampling parameter for controlling diversity of responses (0.0 = deterministic, 1.0 = more diverse)"
    )
    g.add_argument('--top-k', type=int, default=20,
        help="Top-k sampling parameter for controlling diversity of responses (0 = no top-k)"
    )

    g = parser.add_argument_group('Document options')
    g.add_argument('-d', '--documents-folder',
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
    g.add_argument('--llm-as-a-judge',
        action="store_true",
        help="Use LLM-based metrics for answer evaluation"
    )
    g.add_argument('--ollama-address',
        default="http://localhost:11434",
        help="Address of the Ollama server"
    )
    g.add_argument('--ollama-model',
        default="qwen3:8b",
        help="Model name to use for Ollama-based evaluations"
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

    g = parser.add_argument_group('General RAG options')
    g.add_argument('--embeddings-top-k', type=int, help='Number of vector search candidates to retrieve', default=50)

    g = parser.add_argument_group('Hybrid RAG options')
    g.add_argument('--use-bm25-reranker', action='store_true', help='Enable BM25 (keyword-based) reranking', default=False)
    g.add_argument('--bm25-top-k', type=int, help='Number of BM25 candidates to retrieve', default=25)
    g.add_argument('--use-document-reranker', action='store_true', help='Enable document reranking with cross-encoder model', default=False)
    g.add_argument('--hf-document-reranker-model', help='Name of the document reranker model that will be loaded from HuggingFace', default='cross-encoder/ms-marco-MiniLM-L6-v2')
    g.add_argument('--document-reranker-top-n', type=int, help='Number of documents that reranker model should keep', default=10)
    g.add_argument('--document-reranker-score-threshold', type=float, help='Filter reranker results by score threshold. Note that the score scale depends on the model and may range from -10 to 10.', default=None)

    return parser.parse_args()

# Auto mode handler
def process_auto_mode(qa_chain, questions_file, cache_dir, args):  
    # Load questions from JSON file
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    # Initialize result file
    output_path, output_data = initialize_result_file(args, questions_file)
    output_data["categories"] = {}
    
    # Collect all questions and get answers in batch
    all_questions_for_eval = []
    category_question_mapping = []  # To keep track of which question belongs to which category

    # Process each category
    for category, category_data in questions_data['categories'].items():
        print(f"\nProcessing category: {category}")
        output_data['categories'][category] = {"questions": []}

        # Collect questions for this category
        for question_data in category_data['questions']:
            print(f"\nProcessing question: {question_data['question']}")

            # Get answer from AI
            response = qa_chain.invoke({"query": question_data['question']})
            answer = response['result']

            # Create question for evaluation
            all_questions_for_eval.append({
                'question': question_data['question'],
                'answer': answer,
                'reference_answer': question_data['reference_answer'],
                'source_documents': response['source_documents'],
                'weight': question_data['weight'],
            })

            # Store original question data for output
            output_data['categories'][category]['questions'].append({
                "question": question_data['question'],
                "reference_answer": question_data['reference_answer'],
                "weight": question_data['weight'],
                "model_answer": answer,
                "eval_results": {}  # Will be filled after evaluation
            })

            # Keep track of category and question index
            category_question_mapping.append((category, len(output_data['categories'][category]['questions']) - 1))

    # Run batch evaluation for all questions
    print("\nRunning batch evaluation for all questions...")
    eval_results = evaluate_answers(
        all_questions_for_eval,
        verbose=False,  # Don't print results in auto mode
        cache_dir=cache_dir,
        llm_as_a_judge=args.llm_as_a_judge,
        model_name=args.ollama_model
    )

    # Update output data with evaluation results
    if eval_results:
        metrics = eval_results['metrics']
        eval_table = eval_results['eval_table']

        # Update each question's evaluation results
        for i, (category, question_idx) in enumerate(category_question_mapping):
            if i < len(eval_table):
                output_data['categories'][category]['questions'][question_idx]['eval_results'] = {
                    "metrics": metrics,
                    "eval_table": eval_table.iloc[i].to_dict()
                }

        print(f"Overall evaluation metrics: {metrics}")

    # Save final evaluation results
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

            char_limit = min(len(doc.page_content), 150)  # Limit excerpt to 150 characters
            excerpt = doc.page_content[:char_limit].encode('unicode_escape').decode('utf-8')
            similarity_score = doc.metadata.get('similarity_score', 'N/A')
            reranker_score = doc.metadata.get('reranker_score', 'N/A')
            print(f"    Path: {doc_path}")
            print(f"    Excerpt: {excerpt}...")
            print(f"    Similarity score: {similarity_score}")
            print(f"    Relevance score: {reranker_score}\n")

        # Get reference answer for evaluation
        reference_answer = input("\nPlease provide the reference answer for evaluation (or press Enter to skip): ")
        if reference_answer:
            # Create single question for evaluation
            question = [{
                'question': query,
                'answer': answer,
                'reference_answer': reference_answer,
                'source_documents': response['source_documents'],
            }]

            eval_results = evaluate_answers(
                questions=question,
                cache_dir=cache_dir,
                llm_as_a_judge=args.llm_as_a_judge,
                model_name=args.ollama_model
            )
            
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

    # Validate documents folder
    if not os.path.exists(args.documents_folder) or not os.path.isdir(args.documents_folder) :
        print(f"Documents folder {args.documents_folder} does not exist or is not a directory. See the -d or --documents-folder option. Exiting.")
        return

    # List models if requested
    if args.list_models:
        setup_genai_environment()
        validate_model(args.embedding_model)
        return

    # Analyze results if requested
    if args.analyze_results:
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
    configure_mlflow(args.cache_dir, llm_as_a_judge=args.llm_as_a_judge, ollama_address=args.ollama_address)

    # Load and cache document chunks
    chunks, changed = load_and_cache_chunks(args.documents_folder, args.cache_dir)
    print(f"\nDocument chunks loaded. Changes detected: {changed}")

    # Create LLM and build RAG system
    qa_chain = build_rag_system(
        args.embedding_model,
        args.embeddings_top_k,
        args.use_bm25_reranker,
        args.bm25_top_k,
        args.use_document_reranker,
        args.hf_document_reranker_model,
        args.document_reranker_top_n,
        args.document_reranker_score_threshold,
        api_key,
        chunks,
        llm,
        args.cache_dir,
    )

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