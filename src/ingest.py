"""
Step 1: Load PDFs and split them into overlapping text chunks.

Pseudocode:
    load_documents(folder_path) -> load every PDF in the folder (PyPDFLoader)
    split_into_chunks(documents, chunk_size, overlap) ->
        RecursiveCharacterTextSplitter, returns a list of chunk objects
        (each carrying its source document name and page number for later citation)
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(folder_path: str) -> list:
    """Load every PDF in folder_path and return a list of raw document objects."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    pdf_paths = list(folder.glob("*.pdf"))
    if  pdf_paths:
        print(f"Found {len(pdf_paths)} PDF files in folder: {folder_path}")
    else:
        print(f"No PDF files found in folder: {folder_path}")

    documents = []
    for pdf_path in pdf_paths:
       try:
           pages = PyPDFLoader(str(pdf_path)).load()
           documents.extend(pages)
           print(f"Successfully read {len(pages)} pages from {pdf_path.name}")
       except Exception as e:
           print(f"Failed to load {pdf_path.name}: {e}")
      
    return documents
        


def split_into_chunks(documents: list, chunk_size: int = 800, overlap: int = 100) -> list:
    """Split loaded documents into overlapping chunks, preserving source metadata."""
    raise NotImplementedError


if __name__ == "__main__":
    new_documents = load_documents("documents")
    print(f"Total documents loaded: {len(new_documents)}")
    if new_documents:
        print(f"First page content: {new_documents[0].metadata}")
        print(f"First 100 characters of page content: {new_documents[0].page_content[:100]}")
                                     
        