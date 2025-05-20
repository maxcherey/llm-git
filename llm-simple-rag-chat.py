import sys
import os
import time
import argparse
import logging
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import mlflow
import pandas as pd


from llm_simple_rag_chat.genai_utils import setup_genai_environment, validate_model, create_embeddings, create_llm
from llm_simple_rag_chat.document_utils import load_documents, split_documents

exectime_internal = 0.0
exectime_external = 0.0
time_start = time.time()

def build_rag_system(embedding_model, api_key, chunks, llm):
    print(f"Initializing embeddings with model: {embedding_model}")
    
    # Create embeddings instance
    embeddings = create_embeddings(embedding_model, api_key)
    
    # Create a Chroma vector store
    vector_store = Chroma.from_documents(chunks, embeddings)
    print("Vector store created.")
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Define custom prompt
    template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and to the point.
 
Context: {context}
 
Question: {question}
 
Answer:"""
    RAG_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Build RAG chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )

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

    return parser.parse_args()


# GenAI functions moved to genai_utils.py


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

def evaluate_answer(query, answer, reference_answer=None, verbose=True, weight=1.0):
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
            model_type="question-answering",
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

    # Set up logging
    if args.verbose is None:
        level = logging.WARNING
    else:
        level = logging.DEBUG if args.verbose > 1 else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)6s - %(message)s")

    
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