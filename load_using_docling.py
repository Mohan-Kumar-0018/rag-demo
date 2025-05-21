import os
from typing import List
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema.document import Document

def load_documents(source_dir: str) -> List[Document]:
    """Load documents from the source directory using Docling."""
    documents = []
    doc_converter = DocumentConverter()

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if not os.path.isfile(file_path):
            continue

        try:
            if filename.lower().endswith('.pdf'):
                # Load PDF with Docling
                doc_text = doc_converter.convert_file_to_text(file_path)
            elif filename.lower().endswith('.txt'):
                # Load text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_text = f.read()
            else:
                continue

            # Create Document object
            doc = Document(
                page_content=doc_text,
                metadata={"source": filename}
            )
            documents.append(doc)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    return documents

def main():
    # Define paths
    source_dir = "source_documents"
    db_dir = "db2"

    # Create directories if they don't exist
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)

    # Load documents
    print("Loading documents...")
    documents = load_documents(source_dir)
    print(f"Loaded {len(documents)} documents")

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Split documents
    print("Splitting documents...")
    split_docs = text_splitter.split_documents(documents)
    print(f"Created {len(split_docs)} document chunks")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Initialize and load vector store
    print("Creating vector store...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name="docling_docs"
    )

    # Add documents to vector store
    vectorstore.add_documents(split_docs)
    
    # Persist the vector store
    # vectorstore.persist()
    print("Vector store created and persisted successfully")

if __name__ == "__main__":
    main()