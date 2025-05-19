import os
import argparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings



def parse_arguments():
    """Parse command line arguments for model selection."""
    parser = argparse.ArgumentParser(description="RAG application using Google Gemini models")
    parser.add_argument(
        "--reasoning-model", 
        default="models/gemini-2.0-flash",
        help="Gemini model for reasoning/chat"
    )
    parser.add_argument(
        "--embedding-model", 
        default="models/embedding-001",
        help="Gemini model for embeddings"
    )
    parser.add_argument(
        "--documents-folder",
        default="./documents",
        help="Path to the documents folder"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Google models and exit (useful for validating API token and selecting models)"
    )
    return parser.parse_args()


def validate_model(model_name, required_capability):
    """Validate if the specified model is available and has the required capability."""
    available_models = []
    model_found = False
    
    for model in genai.list_models():
        if required_capability in model.supported_generation_methods:
            available_models.append(model.name)
            if model.name == model_name:
                model_found = True
                
    if not model_found:
        print(f"ERROR: Model '{model_name}' not found or doesn't support {required_capability}.")
        print("Available models with this capability:")
        for model in available_models:
            print(f"  {model}")
        return False
    return True


def setup_genai_environment():
    """Set up the environment and validate API key."""
    print("Attempting to load .env...")
    # Load environment variables
    load_dotenv(override=True)  # Ensure it's loaded and overrides existing

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        print("GEMINI_API_KEY loaded successfully!")
        # Set the environment variable for Google embeddings
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        print("GEMINI_API_KEY NOT loaded. Check .env file and path.")
        sys.exit(1)  # Stop if no key

    # Configure genai with the API key
    genai.configure(api_key=api_key)
    
    return api_key


# Parse command line arguments
args = parse_arguments()

# Set up environment and get genai API key
api_key = setup_genai_environment()

# If --list-models is specified, show available models and exit
if args.list_models:
    print("\nAvailable Google models:")
    print("\nModels with generateContent capability:")
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(f"  {model.name}")
    print("\nModels with embedContent capability:")
    for model in genai.list_models():
        if "embedContent" in model.supported_generation_methods:
            print(f"  {model.name}")
    sys.exit(0)

# Validate models
reasoning_model_valid = validate_model(args.reasoning_model, "generateContent")

# Only validate embedding model if it's a Google model
if args.embedding_model.startswith("gemini"):
    embedding_model_valid = validate_model(args.embedding_model, "embedContent")
    if not embedding_model_valid:
        print("Invalid embedding model specified. Exiting.")
        sys.exit(1)
else:
    embedding_model_valid = True
    print("Using HuggingFace embeddings - skipping model validation")

if not reasoning_model_valid:
    print("Invalid reasoning model specified. Exiting.")
    sys.exit(1)


# Initialize the LLM with the validated model
print(f"Initializing LLM with model: {args.reasoning_model}")
llm = ChatGoogleGenerativeAI(
    model=args.reasoning_model,
    temperature=0.1,
    google_api_key=api_key  # Explicitly pass the key
)

# Test the LLM
print("Testing LLM...")
try:
    response = llm.invoke("What is the capital of France?")
    print(f"LLM Response: {response.content}")
except Exception as e:
    print(f"Error initializing or using LLM: {e}")
    sys.exit(1)

# --- 1. Load Documents (Recursively) ---
document_folder = args.documents_folder
documents = []

print(f"Starting to load documents from: {document_folder}")

# Use os.walk to traverse through the main folder and all its subfolders
for dirpath, dirnames, filenames in os.walk(document_folder):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        
        # Determine the correct loader based on file extension
        if filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"Loaded PDF: {file_path}")
            except Exception as e:
                print(f"Error loading PDF {file_path}: {e}")
        elif filename.endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding='utf-8') # Specify encoding for text files
                documents.extend(loader.load())
                print(f"Loaded TXT: {file_path}")
            except Exception as e:
                print(f"Error loading TXT {file_path}: {e}")
        elif filename.endswith(".md") or filename.endswith(".markdown"):
            try:
                loader = UnstructuredMarkdownLoader(file_path)
                documents.extend(loader.load())
                print(f"Loaded Markdown: {file_path}")
            except Exception as e:
                print(f"Error loading Markdown {file_path}: {e}")
        # Add more `elif` conditions here for other document types (e.g., .docx, .html)
        # elif filename.endswith(".docx"):
        #     try:
        #         from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        #         loader = UnstructuredWordDocumentLoader(file_path)
        #         documents.extend(loader.load())
        #         print(f"Loaded DOCX: {file_path}")
        #     except Exception as e:
        #         print(f"Error loading DOCX {file_path}: {e}")
        else:
            print(f"Skipping unsupported file type: {file_path}")


print(f"\nFinished loading. Total documents loaded: {len(documents)}")

# --- 2. Split Documents into Chunks ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# --- 3. Create Embeddings and Store in Vector Store ---
print(f"Initializing embeddings with model: {args.embedding_model}")

# Function to create embeddings based on model type
def create_embeddings(model_name, api_key=None):
    """Create embeddings instance based on model name.
    
    Args:
        model_name (str): Name of the embedding model
        api_key (str, optional): API key for Google embeddings
        
    Returns:
        Embeddings: Instance of the appropriate embeddings class
    """
    if model_name.startswith("models/") or model_name.startswith("gemini"):  # Google embeddings
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            api_key=api_key
        )
    else:  # HuggingFace embeddings
        return HuggingFaceEmbeddings(model_name=model_name)

# Create embeddings instance
embeddings = create_embeddings(args.embedding_model, api_key)

# Create a Chroma vector store (local, in-memory by default, or persistent)
# To make it persistent: persist_directory="./chroma_db"
vector_store = Chroma.from_documents(chunks, embeddings)
print("Vector store created.")

# --- 4. Define the Retriever ---
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

# --- 5. Build the RAG Chain ---
# Define a custom prompt to guide Gemini
template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and to the point.

Context: {context}

Question: {question}

Answer:"""
RAG_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 'stuff' combines all retrieved documents into one context
    retriever=retriever,
    return_source_documents=True, # Optional: return the chunks that were used
    chain_type_kwargs={"prompt": RAG_PROMPT}
)

# --- 6. Answer Questions ---
print("\nReady to answer questions! Type 'exit' to quit.")
while True:
    query = input("Your question: ")
    if query.lower() == 'exit':
        break
    
    response = qa_chain.invoke({"query": query})
    print(f"\nAI's Answer: {response['result']}")
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
                
            # Check for underline-style headers
            if i > 0 and (all(c == '=' for c in line) or all(c == '-' for c in line)):
                section = content_lines[i-1].strip()
                break
                
            # Try to use the first non-empty line as section if nothing else found
            if i == 0 and len(line.strip()) < 100:  # Reasonable section title length
                section = line.strip()
        
        # Print structured information with indentation
        print("    " + "-" * 40)  # Add a shorter separator line with indentation
        print(f"    Path: {doc_path}")
        print(f"    Section: {section}\n")