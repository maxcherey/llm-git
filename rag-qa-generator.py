import os
import time
import argparse
import logging
import json
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from llm_simple_rag_chat.document_utils import (
    load_and_cache_chunks
)

from llm_simple_rag_chat.rag_utils import (
    build_rag_system, create_document_reranker
)

from llm_simple_rag_chat.genai_utils import (
    create_llm,
    create_embeddings,
)

# Suppress TensorFlow and XLA logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

exectime_internal = 0.0
exectime_external = 0.0
time_start = time.time()

def str_presenter(dumper, data):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data"""
    if data.count('\n') > 0:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter) # to use with safe_dum
class VerboseSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple chat application with RAG")

    g = parser.add_argument_group('Generic options')
    g.add_argument('-v', '--verbose', action='count', help='enable verbose mode (use -vv for max verbosity)')
    g.add_argument('-l', '--logfile', action='store', help='log filename')

    g = parser.add_argument_group('Model options')
    g.add_argument('--chat-model-provider',
        default="google",
        help="Chat model provider (e.g., google, huggingface, etc. If using Google, use 'google' for Gemini models)"
    )
    g.add_argument('--chat-model-url',
        default=None,
        help="Base URL for the chat model (if using an external API). If using Google, leave this as None."
    )
    g.add_argument('--chat-model-name',
        default="models/gemini-2.0-flash",
        help="Gemini model for reasoning/chat (e.g., 'models/gemini-2.0-flash' for Gemini 2.0 Flash)"
    )
    g.add_argument('--embedding-model-provider',
        default="google",
        help="Embedding model provider (e.g., google, huggingface, etc. If using Google, use 'google' for Gemini models)"
    )
    g.add_argument('--embedding-model-url',
        default=None,
        help="Base URL for the embedding model (if using an external API). If using Google, leave this as 'None'."
    )
    g.add_argument('--embedding-model-name',
        default="models/embedding-001",
        help="Embedding model name (e.g., 'models/embedding-001' for Google Gemini embeddings, or a HuggingFace model name)"
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
    g.add_argument('--documents-chunk-size', type=int, help='Size of the split document chunk in tokens. Note that maximum chunk size is limited by either embedding model or reranker model.', default=800)
    g.add_argument('--documents-chunk-overlap-size', type=int, help='Size of the chunk overlap size in tokens. Recommended value is between 10-20% of chunk size', default=80)

    g = parser.add_argument_group('Mode options')
    g.add_argument('--questions-file',
        default="questions.json",
        help="Path to the questions JSON file"
    )
    g.add_argument('--cache-dir',
        default=".cache",
        help="Directory to store cached artifacts and data"
    )

    g = parser.add_argument_group('General RAG options')
    g.add_argument('--embeddings-top-k', type=int, help='Number of vector search candidates to retrieve.', default=75)
    g.add_argument('--embeddings-score-threshold', type=float, help='Filter vector search results by similarity score threshold. Vector storage uses cosine similarity where with the scale from 0 (different) to 1 (similar). Not used if set to "None".', default=None)

    g = parser.add_argument_group('Hybrid RAG options')
    g.add_argument('--use-bm25-reranker', action='store_true', help='Enable BM25 (keyword-based) reranking.', default=False)
    g.add_argument('--bm25-top-k', type=int, help='Number of BM25 candidates to retrieve', default=50)
    g.add_argument('--bm25-weight', type=float, help='Weight of BM25 candidates.', default=0.5)
    g.add_argument('--document-reranker-provider', type=str, help='Enable document reranking with cross-encoder model. Not used if not specified', default=None)
    g.add_argument('--document-reranker-model', help='Name of the document reranker model that will be loaded from HuggingFace', default='cross-encoder/ms-marco-MiniLM-L6-v2')
    g.add_argument('--document-reranker-url', help='URL of the external reranker model, for example "http://127.0.0.1/v1/rerank".', default=None)
    g.add_argument('--document-reranker-api-token', help='API token for the external reranker model.', default=None)
    g.add_argument('--document-reranker-top-n', type=int, help='Number of documents that reranker model should keep', default=10)
    g.add_argument('--document-reranker-normalize-scores', action=argparse.BooleanOptionalAction, help='Apply Sigmoid function that normalizes scores to 0 (irrelevant) to 1 (relevant).', default=True)
    g.add_argument('--document-reranker-score-threshold', type=float, help='Filter reranker results by relevance score threshold. Not used if set to "None".', default=None)

    return parser.parse_args()

def process_auto_mode(qa_chain, questions_file):
    # Load questions from JSON file
    with open(questions_file, 'r', encoding='utf-8') as f:
        doc_questions = json.load(f)

    collection = []

    # Save all collected questions and answers to a JSON file
    answers_dir = '.answers'
    os.makedirs(answers_dir, exist_ok=True)
    output_file = os.path.join(answers_dir, f"answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")

    # Collect questions for this category
    for doc, questions in doc_questions.items():
        print(f"\nProcessing questions for document '{doc}'")

        total_questions = len(questions)
        for i, question in enumerate(questions, start=1):
            print(f"Question {i}/{total_questions}: {question}")

            # Check if the question is empty
            if not question.strip():
                print("Skipping empty question.")
                continue

            # Get answer from AI
            response = qa_chain.invoke({"query": question + "/no_think"}) # TODO: Required for Qwen3
            answer = response['result'].replace('<think>', '').replace('</think>', '').strip() # TODO: Required for Qwen3
            # TODO: Maybe filter by keywords like "I don't know" or "I do not know"?
            collection.append({
                "question": question,
                "answer": answer,
                "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in response['source_documents']],
            })
            print(f"Answer: {answer}\n")
        print("Writing intermediate results to file...")
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(collection, f, indent=2, allow_unicode=True, sort_keys=False, Dumper=VerboseSafeDumper)
    print(f"All questions processed. Answers saved to {output_file}")


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

    # Validate documents folder
    if not os.path.exists(args.documents_folder) or not os.path.isdir(args.documents_folder) :
        print(f"Documents folder {args.documents_folder} does not exist or is not a directory. See the -d or --documents-folder option. Exiting.")
        return

    # Ensure cache directory exists
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created cache directory at: {cache_dir}")

    # Setup an LLM
    llm = create_llm(
        args.chat_model_provider,
        args.chat_model_name,
        args.chat_model_url,
        args.temperature,
        args.n_tokens,
        args.top_p,
        args.top_k
    )

    # Load and cache document chunks
    # TODO: Use embedder tokenizer
    chunks, changed = load_and_cache_chunks(
        args.documents_folder,
        args.documents_chunk_size,
        args.documents_chunk_overlap_size,
        llm.get_num_tokens,
        args.cache_dir
    )
    print(f"\nDocument chunks loaded. Changes detected: {changed}")

    # Use the correct argument name for embedding model
    embeddings = create_embeddings(args.embedding_model_provider, args.embedding_model_name, args.embedding_model_url)

    reranker = create_document_reranker(
        args.document_reranker_provider,
        args.document_reranker_model,
        args.document_reranker_url,
        args.document_reranker_top_n,
        args.document_reranker_normalize_scores,
        args.document_reranker_score_threshold
    )

    # Create LLM and build RAG system
    qa_chain = build_rag_system(
        llm,
        embeddings,
        args.embeddings_top_k,
        args.embeddings_score_threshold,
        args.use_bm25_reranker,
        args.bm25_weight,
        args.bm25_top_k,
        reranker,
        chunks,
        args.cache_dir,
    )

    process_auto_mode(qa_chain, args.questions_file)

if __name__ == "__main__":
    main()