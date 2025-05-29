# RAG Demo

A demonstration of Retrieval-Augmented Generation (RAG) using LangChain, HuggingFace embeddings, Chroma vector database, and Ollama.

## Overview

This repository demonstrates how to build a simple RAG system that:

1. Loads documents from a directory (PDF and text files)
2. Processes and splits them into manageable chunks
3. Creates embeddings using HuggingFace's sentence transformer model
4. Stores these embeddings in a Chroma vector database
5. Allows querying the knowledge base with natural language questions
6. Generates contextually relevant answers using Ollama LLM

## Requirements

- Python 3.8+
- LangChain
- HuggingFace transformers
- Chroma DB
- Ollama (running locally)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Install Ollama following instructions at [ollama.com](https://ollama.com)
2. Pull the desired Llama model:

```bash
ollama pull llama3.2:latest
```

3. Place your documents in the `source_documents` directory:
   - Supported formats: PDF, TXT

## Usage

### 1. Process Documents

Load and process documents to create the vector database:

```bash
python load_docs.py
```

Options:
- `--verbose`: Enable verbose debug output

This will:
- Load all documents from the `source_documents` directory
- Split them into chunks (default size: 1000 characters with 200 character overlap)
- Create embeddings using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model
- Store the vector database in the `db` directory

### 2. Query the Knowledge Base

After processing documents, you can query the system:

```bash
python query_docs.py "Your question here?"
```

Options:
- `--verbose`: Enable verbose output
- `--top-k N`: Number of documents to retrieve (default: 4)

This will:
- Retrieve the most relevant document chunks based on your question
- Generate a response using the Ollama LLM, incorporating the context from retrieved documents

## Environment Variables

- `OLLAMA_MODEL`: Specify which Ollama model to use (default: `llama3.2:latest`)

## Project Structure

- `load_docs.py`: Script to load documents and create the vector database
- `query_docs.py`: Script to query the knowledge base
- `source_documents/`: Directory to store your PDF and text files
- `db/`: Directory where the Chroma vector database is stored
- `requirements.txt`: Python dependencies

## How It Works

1. **Document Loading**: The system uses LangChain's document loaders to read PDF and text files.
2. **Document Splitting**: Documents are split into smaller chunks for more effective retrieval.
3. **Embedding Generation**: Text chunks are converted to vector embeddings using HuggingFace's sentence transformer.
4. **Vector Storage**: Embeddings are stored in a Chroma vector database for efficient similarity search.
5. **Query Processing**: When a question is asked, it's converted to an embedding and used to find the most similar document chunks.
6. **Response Generation**: Retrieved chunks are used as context for the Ollama LLM to generate a relevant and informed response.

