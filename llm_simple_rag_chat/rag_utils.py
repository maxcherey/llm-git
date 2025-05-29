import os
import json
import hashlib
import operator
from pydantic import Field
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from llm_simple_rag_chat.genai_utils import create_embeddings
from langchain_core.runnables import chain

class ScoredCrossEncoderReranker(CrossEncoderReranker):
    score_threshold: float | None = Field(default=None)

    def compress_documents(self, documents, query, callbacks = None):
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        out = []
        for doc, score in result[: self.top_n]:
            if self.score_threshold is None or score >= self.score_threshold:
                doc.metadata['reranker_score'] = score
                out.append(doc)
        return out

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

def build_rag_system(
    embedding_model,
    embeddings_top_k: int,
    use_bm25_reranker: bool,
    bm25_top_k: int,
    use_document_reranker: bool,
    hf_document_reranker_model: str,
    document_reranker_top_n: int,
    document_reranker_score_threshold: float,
    api_key,
    chunks,
    llm,
    cache_dir=".cache"
):
    print(f"Initializing embeddings with model: {embedding_model}")

    # Create embeddings instance
    embeddings = create_embeddings(embedding_model, api_key)

    # Get or create vector store
    vector_store = get_cached_vector_store(embedding_model, chunks, embeddings, cache_dir)

    @chain
    def vector_retriever(query: str):
        docs, scores = zip(*vector_store.similarity_search_with_score(query, embeddings_top_k))
        for doc, score in zip(docs, scores):
            doc.metadata["similarity_score"] = score
        return docs

    retrievers = [vector_retriever]
    # TODO: Configurable weights for retrievers
    weights = [0.25]
    if use_bm25_reranker:
        print('Using BM25 retriever with top_k:', bm25_top_k)
        retrievers += [BM25Retriever.from_documents(chunks, search_kwargs={"k": bm25_top_k})]
        weights += [0.75]

    # Create ensembled retriever
    base_retriever = EnsembleRetriever(
        retrievers=retrievers,
        weights=weights
    )

    if use_document_reranker:
        print('Using document reranker:', hf_document_reranker_model, 'with top_n:', document_reranker_top_n)
        model = HuggingFaceCrossEncoder(model_name=hf_document_reranker_model, model_kwargs={'device': 'cpu'})
        compressor = ScoredCrossEncoderReranker(score_threshold=document_reranker_score_threshold, model=model, top_n=document_reranker_top_n)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
    else:
        retriever = base_retriever

    # Define custom prompt
    template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

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
