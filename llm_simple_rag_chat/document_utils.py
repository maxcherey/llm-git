import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(document_folder):
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
                    loaded_docs = loader.load()
                    if loaded_docs is not None:
                        documents.extend(loaded_docs)
                    print(f"Loaded PDF: {file_path}")
                except Exception as e:
                    print(f"Error loading PDF {file_path}: {e}")
            elif filename.endswith(".txt"):
                try:
                    loader = TextLoader(file_path, encoding='utf-8')  # Specify encoding for text files
                    loaded_docs = loader.load()
                    if loaded_docs is not None:
                        documents.extend(loaded_docs)
                    print(f"Loaded TXT: {file_path}")
                except Exception as e:
                    print(f"Error loading TXT {file_path}: {e}")
            elif filename.endswith(".md") or filename.endswith(".markdown"):
                try:
                    loader = UnstructuredMarkdownLoader(file_path)
                    loaded_docs = loader.load()
                    if loaded_docs is not None:
                        documents.extend(loaded_docs)
                    print(f"Loaded Markdown: {file_path}")
                except Exception as e:
                    print(f"Error loading Markdown {file_path}: {e}")
            # Add more `elif` conditions here for other document types (e.g., .docx, .html)
            # elif filename.endswith(".docx"):
            #     try:
            #         from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            #         loader = UnstructuredWordDocumentLoader(file_path)
            #         loaded_docs = loader.load()
            #         if loaded_docs is not None:
            #             documents.extend(loaded_docs)
            #         print(f"Loaded DOCX: {file_path}")
            #     except Exception as e:
            #         print(f"Error loading DOCX {file_path}: {e}")
            else:
                print(f"Skipping unsupported file type: {file_path}")
    
    print(f"\nFinished loading. Total documents loaded: {len(documents)}")
    return documents if documents else []

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
