import os
import sys
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define paths
SOURCE_DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "source_documents")
DB_DIR = os.path.join(os.path.dirname(__file__), "db")

# Function to load documents from a directory
def load_documents(source_dir: str) -> List:
    """
    Load documents from the specified directory.
    
    Args:
        source_dir: Directory containing documents to load
    
    Returns:
        List of loaded documents
    """
    print(f"Loading documents from {source_dir}")
    
    # Create a loader for text files
    loader = DirectoryLoader(
        source_dir,
        glob="**/*.txt",  # Load all .txt files
        loader_cls=TextLoader,
        show_progress=True
    )
    
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

# Function to split documents into chunks
def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """
    Split documents into chunks for processing.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of document chunks
    """
    print(f"Splitting {len(documents)} documents into chunks")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

# Function to create embeddings and store in Chroma DB
def create_vector_store(chunks: List, db_dir: str):
    """
    Create embeddings and store in a vector database.
    
    Args:
        chunks: Document chunks to embed
        db_dir: Directory to store the vector database
    
    Returns:
        Chroma vector store instance
    """
    print(f"Creating vector store in {db_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)
    
    # Initialize the HuggingFace embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_dir
    )
    
    print(f"Vector store created successfully")
    return vector_store

# Main function to run the document loading process
def main():
    # Check if the source directory exists
    if not os.path.exists(SOURCE_DOCUMENTS_DIR):
        print(f"Error: Source directory '{SOURCE_DOCUMENTS_DIR}' does not exist.")
        sys.exit(1)
        
    # Load documents
    documents = load_documents(SOURCE_DOCUMENTS_DIR)
    
    # Split documents into chunks
    chunks = split_documents(documents)
    
    # Create vector store
    vector_store = create_vector_store(chunks, DB_DIR)
    
    print("Document loading and processing complete")

if __name__ == "__main__":
    main()
