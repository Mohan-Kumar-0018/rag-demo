import os
import sys
import logging
import argparse
from typing import List, Dict, Any, Union

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.document_loaders.helpers import detect_file_encodings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define paths
SOURCE_DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "source_documents")
DB_DIR = os.path.join(os.path.dirname(__file__), "db")

# Function to get the appropriate document loader based on file extension
def get_loader_class(file_path: str):
    """
    Determine the appropriate document loader based on file extension
    
    Args:
        file_path: Path to the file
        
    Returns:
        Appropriate loader class for the file type
    """
    _, file_extension = os.path.splitext(file_path.lower())
    
    if file_extension == '.pdf':
        return PyPDFLoader
    elif file_extension == '.txt':
        return TextLoader
    else:
        # Default to TextLoader for unknown types
        logger.debug(f"Unknown file extension '{file_extension}'. Defaulting to TextLoader.")
        return TextLoader

# Function to load documents from a directory
def load_documents(source_dir: str) -> List:
    """
    Load documents from the specified directory.
    
    Args:
        source_dir: Directory containing documents to load
    
    Returns:
        List of loaded documents
    """
    logging.info(f"Loading documents from {source_dir}")
    
    # Load all documents, combining results from different document types
    all_documents = []
    
    # Process PDF files
    pdf_loader = DirectoryLoader(
        source_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    pdf_documents = pdf_loader.load()
    logging.info(f"Loaded {len(pdf_documents)} PDF documents")
    all_documents.extend(pdf_documents)
    
    # Process text files
    txt_loader = DirectoryLoader(
        source_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    txt_documents = txt_loader.load()
    logging.info(f"Loaded {len(txt_documents)} text documents")
    all_documents.extend(txt_documents)
    
    logging.info(f"Loaded {len(all_documents)} total documents")
    return all_documents

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
    logging.info(f"Splitting {len(documents)} documents into chunks")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    logging.debug(f"Split into {len(chunks)} chunks")
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
    logging.info(f"Creating vector store in {db_dir}")
    
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
    
    logging.info(f"Vector store created successfully")
    return vector_store

# Main function to run the document loading process
def main():
    """Main function to run the document loading process"""
    parser = argparse.ArgumentParser(description='Load and process documents for RAG')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose debug output')
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    
    try:
        # Load documents from source directory
        documents = load_documents(SOURCE_DOCUMENTS_DIR)
        logger.info(f"Loaded {len(documents)} documents from {SOURCE_DOCUMENTS_DIR}")
        
        # Split documents into chunks
        chunks = split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks")
        
        # Create vector store
        vector_store = create_vector_store(chunks, DB_DIR)
        logger.info(f"Created vector store at {DB_DIR}")
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
