import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

def setup_genai_environment():
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

    genai.configure(api_key=api_key)
    
    return api_key


def validate_model(model_name, required_capability):
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


# Function to create embeddings based on model type
def create_embeddings(model_name, api_key=None):
    if model_name.startswith("models/") or model_name.startswith("gemini"):  # Google embeddings
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            api_key=api_key
        )
    else:  # HuggingFace embeddings
        return HuggingFaceEmbeddings(model_name=model_name)


def create_llm(args):
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

    return api_key, llm