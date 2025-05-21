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
class OllamaLLM:
    """
    A class to interact with Ollama API for model inference
    """
    def __init__(self, model_name="llama3.2:latest", api_base="http://localhost:11434"):
        """
        Initialize the Ollama LLM client
        
        Args:
            model_name: Name of the model in Ollama (e.g., 'llama3')
            api_base: Base URL for the Ollama API
        """
        self.model_name = model_name
        self.api_base = api_base
        self.api_url = f"{api_base}/api/generate"
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{api_base}/api/tags")
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json()["models"]]
                print(f"Available models in Ollama: {', '.join(available_models)}")
                
                # Check if our model exists
                if not any(model_name.split(":")[0] in model for model in available_models):
                    print(f"Warning: Model '{model_name}' not found in available models.")
                    print(f"You may need to pull it first with: ollama pull {model_name}")
            else:
                print(f"Warning: Couldn't fetch model list from Ollama (status code: {response.status_code})")
        except requests.RequestException as e:
            print(f"Warning: Couldn't connect to Ollama at {api_base}: {e}")
            print("Please ensure Ollama is running.")
    
    def __call__(self, prompt, max_tokens=1024, temperature=0.7, top_p=0.9, repeat_penalty=1.1, echo=False):
        """
        Call the Ollama API to generate a response
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repeat_penalty: Penalty for repeating tokens
            echo: Whether to echo the prompt in the response
            
        Returns:
            dict: Response containing generated text and metadata
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
            
            response = requests.post(self.api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Format the response similar to what llama-cpp-python returns
                return {
                    "choices": [
                        {
                            "text": result["response"],
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    }
                }
            else:
                print(f"Error from Ollama API: {response.status_code}")
                print(response.text)
                return {"choices": [{"text": "Error generating response from Ollama.", "finish_reason": "error"}]}
                
        except Exception as e:
            print(f"Error calling Ollama API: {e}")
            return {"choices": [{"text": f"Error: {str(e)}", "finish_reason": "error"}]}

# Initialize the Ollama LLM client
def get_llm():
    """
    Initialize the Ollama client for Llama 3.2 model
    
    Returns:
        OllamaLLM: Initialized Ollama client instance
    """
    # Get model name from environment variable or use default
    model_name = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
    
    print(f"Using Ollama model: {model_name}")
    
    # Create and return the Ollama client
    return OllamaLLM(model_name=model_name)

# Build RAG prompt template
def get_rag_prompt():
    """
    Create a prompt template for RAG
    
    Returns:
        PromptTemplate: The prompt template for RAG
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
    
    return PromptTemplate.from_template(template)

# Process query using RAG
def process_query(query: str, vector_store: Chroma, llm, top_k: int = 4):
    """
    Process a query using RAG
    
    Args:
        query: User query
        vector_store: Vector store for document retrieval
        llm: LLM for generating responses
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
    prompt = get_rag_prompt()
    
    print(f"Formatted prompt...{prompt.template}")
    print(f"Retrieved {len(docs)} documents for context")
    print(f"Context: {context_text[:100]}...")  # Print first 100 characters of context
    print(f"Query: {query}")
    
    # Format the prompt with the retrieved context and user query
    formatted_prompt = prompt.format(context=context_text, query=query)
    
    print("Generating response using LLM...")
    
    # Generate response using LLM
    response = llm(
        formatted_prompt,
        max_tokens=1024,
        temperature=0.5,
        top_p=0.9,
        repeat_penalty=1.1,
        echo=False
    )
    
    # Extract the generated text
    answer = response["choices"][0]["text"].strip()
    
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
    parser = argparse.ArgumentParser(description="Query documents using RAG")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--top_k", type=int, default=4, help="Number of documents to retrieve")
    args = parser.parse_args()
    
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