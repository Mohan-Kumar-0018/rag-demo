import os
import sys
import argparse
import json
import requests
from typing import List, Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

# Define paths
DB_DIR = os.path.join(os.path.dirname(__file__), "db")

# Initialize the HuggingFace embeddings - same as in load_docs.py
def get_embeddings():
    """
    Initialize the embedding model using the same configuration as load_docs.py
    
    Returns:
        HuggingFaceEmbeddings: Initialized embedding model
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

# Load the Chroma vector store
def load_vector_store(db_dir: str):
    """
    Load the existing Chroma vector store from disk
    
    Args:
        db_dir: Directory containing the vector database
    
    Returns:
        Chroma: Loaded vector store instance
    """
    print(f"Loading vector store from {db_dir}")
    
    embedding_model = get_embeddings()
    
    # Load the existing vector store
    vector_store = Chroma(
        persist_directory=db_dir,
        embedding_function=embedding_model
    )
    
    print(f"Vector store loaded successfully")
    return vector_store

# Class for interacting with Ollama API
class OllamaAPI:
    """
    A class to interact with Ollama API for model inference
    """
    def __init__(self, model_name="llama3:latest", api_base="http://localhost:11434"):
        """
        Initialize the Ollama API client
        
        Args:
            model_name: Name of the model in Ollama (default: llama3:latest)
            api_base: Base URL for the Ollama API
        """
        self.model_name = model_name
        self.api_base = api_base
        self.generate_url = f"{api_base}/api/generate"
        self.models_url = f"{api_base}/api/tags"
        
        # Check if Ollama is running and if the model exists
        self._check_ollama_status()
        
    def _check_ollama_status(self):
        """
        Check if Ollama is running and if the requested model exists
        """
        try:
            response = requests.get(self.models_url)
            if response.status_code == 200:
                models_data = response.json()
                if "models" in models_data:
                    available_models = [model["name"] for model in models_data["models"]]
                    model_base_name = self.model_name.split(":")[0]
                    
                    print(f"Available models in Ollama: {', '.join(available_models)}")
                    
                    if not any(model_base_name in model for model in available_models):
                        print(f"Warning: Model '{self.model_name}' not found in available models.")
                        print(f"You may need to pull it with: ollama pull {self.model_name}")
                else:
                    print("No models found in Ollama.")
            else:
                print(f"Warning: Couldn't fetch model list. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Warning: Couldn't connect to Ollama at {self.api_base}: {e}")
            print("Please ensure Ollama is running.")
    
    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, 
                top_p: float = 0.9, repeat_penalty: float = 1.1) -> Dict[str, Any]:
        """
        Generate text using the Ollama API
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative/random)
            top_p: Top-p sampling parameter
            repeat_penalty: Penalty for repeating tokens
        
        Returns:
            Dict: Response containing generated text
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty
                }
            }
            
            print(f"Generating response using Ollama model: {self.model_name}")
            response = requests.post(self.generate_url, json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error from Ollama API: {response.status_code}, {response.text}")
                return {"response": f"Error: API returned status code {response.status_code}"}
        
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return {"response": f"Error: {str(e)}"}

# Get the LLM using Ollama
def get_llm():
    """
    Initialize Ollama API client for the Llama model
    
    Returns:
        OllamaAPI: Initialized Ollama API client
    """
    # Get model name from environment variable or use default
    model_name = os.environ.get("OLLAMA_MODEL", "llama3:latest")
    
    print(f"Initializing Ollama API client for model: {model_name}")
    return OllamaAPI(model_name=model_name)

# Build RAG prompt template
def get_rag_prompt():
    """
    Create a prompt template for RAG
    
    Returns:
        str: The prompt template for RAG
    """
    template = """
You are an assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Please provide a helpful, accurate, and concise answer based on the provided context.
If the context doesn't contain information to answer the question, say so rather than making up an answer.

Answer:
"""
    
    return template

# Process query using RAG
def process_query(query: str, vector_store: Chroma, llm: OllamaAPI, top_k: int = 4):
    """
    Process a query using RAG
    
    Args:
        query: User query
        vector_store: Vector store for document retrieval
        llm: Ollama API client for generating responses
        top_k: Number of documents to retrieve
    
    Returns:
        Dict[str, Any]: Query results including answer and retrieved documents
    """
    print(f"Processing query: {query}")
    
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=top_k)
    
    # Prepare context from retrieved documents
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # Get the prompt template
    prompt_template = get_rag_prompt()
    
    # Format the prompt with the retrieved context and user query
    formatted_prompt = prompt_template.format(context=context_text, query=query)
    
    # Generate response using LLM
    response = llm.generate(formatted_prompt)
    
    # Extract the generated text
    answer = response.get("response", "Error: No response generated")
    
    # Prepare the result
    result = {
        "query": query,
        "answer": answer,
        "documents": [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in docs
        ]
    }
    
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query documents using RAG with Ollama")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--top_k", type=int, default=4, help="Number of documents to retrieve")
    parser.add_argument("--model", type=str, help="Ollama model name (default: llama3:latest or from OLLAMA_MODEL env var)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0 to 1.0)")
    args = parser.parse_args()
    
    # Set model if provided as argument
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    
    # Get query from command line or prompt
    query = args.query
    if not query:
        query = input("Enter your query: ")
    
    # Check if the DB directory exists
    if not os.path.exists(DB_DIR):
        print(f"Error: Vector database directory '{DB_DIR}' does not exist.")
        print("Please run load_docs.py first to create the vector database.")
        sys.exit(1)
    
    # Load vector store
    vector_store = load_vector_store(DB_DIR)
    
    # Initialize LLM
    llm = get_llm()
    
    # Process query
    result = process_query(query, vector_store, llm, top_k=args.top_k)
    
    # Print the answer
    print("\nAnswer:")
    print("-------")
    print(result["answer"])
    
    # Print the sources
    print("\nSources:")
    print("--------")
    for i, doc in enumerate(result["documents"]):
        print(f"Source {i+1}: {doc['metadata'].get('source', 'Unknown')}")

if __name__ == "__main__":
    main()