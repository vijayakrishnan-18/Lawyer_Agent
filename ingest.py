import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

# Configuration
VECTOR_DB_PATH = "./chroma_db"
DATA_PATH = "./legal_docs"  # Put your BNS2023.pdf here

def ingest_documents():
    """
    Phase 1: Data Ingestion
    Loads legal PDFs, chunks them preserving context, and stores in Vector DB.
    """
    print("--- STARTED INGESTION ---")
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Please place your PDF files in the '{DATA_PATH}' folder and run this again.")
        return
    
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"Loading: {file}")
            try:
                loader = PyPDFLoader(os.path.join(DATA_PATH, file))
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file}: {e}")

    if not documents:
        print("No documents found.")
        return

    # RESEARCH-GRADE CHUNKING:
    # We use a large chunk size with overlap to ensure legal context (sections) 
    # isn't cut off.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "Section", "CHAPTER", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from documents.")

    # EMBEDDINGS:
    # Using a local, free, high-performance model suitable for legal text
    print("Creating Vector Store (this may take a minute)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_PATH
    )
    print("--- INGESTION COMPLETE ---")

if __name__ == "__main__":
    ingest_documents()
     #Create embeddings and store in Chroma Vector DB  