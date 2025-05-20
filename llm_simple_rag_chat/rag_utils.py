from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from llm_simple_rag_chat.genai_utils import create_embeddings

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
