import os
import json
import hashlib
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from genai_utils import create_embeddings

def get_cached_vector_store(embedding_model, chunks, embeddings, cache_dir):
    vector_store_path = os.path.join(cache_dir, f"vector_store_{embedding_model}")
    state_file_path = os.path.join(vector_store_path, "vector_state.json")
    
    # Create hash of chunks using hashlib
    chunks_data = [(chunk.page_content, str(chunk.metadata)) for chunk in chunks]
    chunks_str = str(chunks_data).encode('utf-8')
    chunks_hash = hashlib.sha256(chunks_str).hexdigest()
    
    # First check if state file exists and matches
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, 'r') as f:
                state = json.load(f)
            if state.get('embedding_model') == embedding_model and state.get('chunks_hash') == chunks_hash:
                # Try to load vector store
                vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
                print("Loaded existing vector store from disk")
                return vector_store
        except Exception as e:
            print(f"Error loading cached vector store: {e}")
            
    # If we got here, either state file doesn't exist or state doesn't match
    print("State mismatch detected, creating new vector store")
    # Create new vector store and persist it
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings,
        persist_directory=vector_store_path
    )
    print("Vector store created and persisted to disk.")
    
    # Save state file
    state = {
        'embedding_model': embedding_model,
        'chunks_hash': chunks_hash
    }
    os.makedirs(vector_store_path, exist_ok=True)
    with open(state_file_path, 'w') as f:
        json.dump(state, f)
    
    return vector_store

def build_rag_system(embedding_model, api_key, chunks, llm, cache_dir=".cache"):
    print(f"Initializing embeddings with model: {embedding_model}")
    
    # Create embeddings instance
    embeddings = create_embeddings(embedding_model, api_key)
    
    # Get or create vector store
    vector_store = get_cached_vector_store(embedding_model, chunks, embeddings, cache_dir)
    
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
