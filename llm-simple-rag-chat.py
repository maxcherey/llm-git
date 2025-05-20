import sys
import os
import time
import argparse
import logging
import json
from pathlib import Path
from llm_simple_rag_chat.genai_utils import setup_genai_environment, validate_model, create_llm
from llm_simple_rag_chat.document_utils import load_documents, split_documents
from llm_simple_rag_chat.rag_utils import build_rag_system
from llm_simple_rag_chat.eval_utils import evaluate_answer

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
def process_auto_mode(qa_chain, questions_file):
    import json
    
    # Load questions from JSON file
    with open(questions_file, 'r') as f:
        data = json.load(f)
    
    # Process each question
    for question_data in data['questions']:
        print(f"\nProcessing question: {question_data['question']}")
        
        # Get answer from AI
        response = qa_chain.invoke({"query": question_data['question']})
        answer = response['result']
        
        # Update the model_answer field
        question_data['model_answer'] = answer
        
        # Run evaluation
        eval_results = evaluate_answer(
            question_data['question'],
            answer,
            question_data['reference_answer'],
            verbose=False,  # Don't print results in auto mode
            weight=question_data['weight']
        )
        
        if eval_results:
            # Update eval_results with the evaluation metrics
            question_data['eval_results'] = {
                "metrics": eval_results['metrics'],
                "eval_table": eval_results['eval_table'].to_dict()
            }
            
            # Print summary metrics
            metrics = eval_results['metrics']
            print(f"Evaluation metrics: {metrics}")
        
        # Save progress after each question
        with open(questions_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Answer and evaluation saved for question {question_data['question'][:50]}...")
        print(f"Evaluation metrics: {metrics}")

    print("\nAll questions processed successfully!")

def run_interactive_mode(qa_chain):
    print("\nReady to answer questions! Type 'exit' to quit.")
    while True:
        query = input("Your question: ")
        if query.lower() == 'exit':
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
            evaluate_answer(query, answer, reference_answer)


def main():
    args = parse_arguments()
    
    # Ensure cache directory exists
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created cache directory at: {cache_dir}")
    
    # Set up logging
    if args.verbose is None:
        level = logging.WARNING
    else:
        level = logging.DEBUG if args.verbose > 1 else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)6s - %(message)s")

    # Setup Google Generative AI environment
    setup_genai_environment()

    api_key, llm = create_llm(args)

    # If list models flag is set, exit after listing models
    if args.list_models:
        return

    # Load and process documents
    documents = load_documents(args.documents_folder)
    chunks = split_documents(documents)
    
    # Create LLM and build RAG system

    qa_chain = build_rag_system(args.embedding_model, api_key, chunks, llm)

    # Run in selected mode
    if args.mode == "interactive":
        run_interactive_mode(qa_chain)
    else:
        process_auto_mode(qa_chain, args.questions_file)

    # Log execution time
    time_end = time.time()
    logging.info(f"Execution time: {time_end - time_start:.2f} seconds")

if __name__ == "__main__":
    main()