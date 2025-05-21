import os
import json
import logging
import argparse
from typing import List, Dict, Any

import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configure logging
def setup_logging(debug: bool = False):
    """Configure logging with optional debug level"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    return logger

# Create a logger instance
logger = setup_logging()

# Define paths
DB_DIR = os.path.join(os.path.dirname(__file__), "db2")

def load_vector_store(db_dir: str) -> Chroma:
    """
    Load the vector store from the specified directory.
    
    Args:
        db_dir: Directory containing the vector database
        
    Returns:
        Chroma vector store instance
    """
    logger.info(f"Loading vector store from {db_dir}")
    
    # Initialize the same embedding model as used in load_using_docling.py
    embedding_model = "sentence-transformers/all-mpnet-base-v2"
    logger.info(f"Initializing HuggingFaceEmbeddings with model: {embedding_model}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
    )
    
    # Load the existing vector store
    vector_store = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings
    )
    
    collection = vector_store._collection
    count = collection.count()
    logger.info(f"Loaded vector store with {count} documents")
    
    return vector_store

def similarity_search(vector_store: Chroma, query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Perform similarity search on the vector store.
    
    Args:
        vector_store: Loaded Chroma vector store
        query: Search query string
        k: Number of results to return
        
    Returns:
        List of documents with their metadata
    """
    logger.info(f"Performing similarity search for query: {query}")
    
    # Perform the similarity search
    results = vector_store.similarity_search_with_relevance_scores(
        query,
        k=k
    )
    
    # Format results
    formatted_results = []
    for doc, score in results:
        result = {
            'content': doc.page_content,
            'metadata': doc.metadata,
            'relevance_score': score
        }
        formatted_results.append(result)
        
        # Log preview of each result
        content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        logger.debug(f"Result (score: {score:.4f}): {content_preview}")
        logger.debug(f"Metadata: {doc.metadata}")
    
    logger.info(f"Found {len(formatted_results)} relevant documents")
    return formatted_results

def query_ollama(query: str, context: str, model: str = "llama2:latest") -> str:
    """
    Query the Ollama API with the given query and context.
    
    Args:
        query: User query
        context: Retrieved context from vector store
        model: Ollama model to use
        
    Returns:
        Generated response from the model
    """
    logger.info(f"Querying Ollama with model: {model}")
    
    # Construct the prompt
    prompt = f"""Based on the following context, please answer the question. If you cannot answer the question based on the context, please say so.

Context:
{context}

Question: {query}

Answer:"""
    
    # Make request to Ollama API
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        result = response.json()
        return result["response"]
    except Exception as e:
        logger.error(f"Error querying Ollama: {str(e)}")
        return f"Error: Failed to get response from Ollama - {str(e)}"

def process_query(query: str, vector_store: Chroma, model: str = "llama2:latest", top_k: int = 4, show_context: bool = False) -> Dict[str, Any]:
    """
    Process a query using RAG workflow.
    
    Args:
        query: User query
        vector_store: Loaded Chroma vector store
        model: Ollama model to use
        top_k: Number of documents to retrieve
        show_context: Whether to include context in response
        
    Returns:
        Dictionary containing response and optionally context
    """
    # Get relevant documents
    docs = similarity_search(vector_store, query, k=top_k)
    
    # Combine document contents for context
    context = "\n\n".join([doc["content"] for doc in docs])
    
    # Get response from Ollama
    response = query_ollama(query, context, model)
    
    result = {"response": response}
    if show_context:
        result["context"] = docs
    
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query documents using RAG with Ollama")
    parser.add_argument("query", help="Query to process")
    parser.add_argument("--model", default="llama2:latest", help="Ollama model to use")
    parser.add_argument("--top_k", type=int, default=4, help="Number of documents to retrieve")
    parser.add_argument("--show-context", action="store_true", help="Show retrieved context")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.debug)
    
    logger.info("=== Starting RAG query processing ===")
    
    # Load the vector store
    vector_store = load_vector_store(DB_DIR)
    
    # Process the query
    result = process_query(
        query=args.query,
        vector_store=vector_store,
        model=args.model,
        top_k=args.top_k,
        show_context=args.show_context
    )
    
    # Print the response
    print("\nResponse:")
    print(result["response"])
    
    if args.show_context:
        print("\nRetrieved Context:")
        for i, doc in enumerate(result["context"], 1):
            print(f"\nDocument {i} (Score: {doc['relevance_score']:.4f}):")
            print(doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"])
            print(f"Metadata: {doc['metadata']}")

if __name__ == "__main__":
    main()
