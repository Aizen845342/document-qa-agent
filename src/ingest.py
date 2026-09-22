"""
Step 1: Load PDFs and split them into overlapping text chunks.

Pseudocode:
    load_documents(folder_path) -> load every PDF in the folder (PyPDFLoader)
    split_into_chunks(documents, chunk_size, overlap) ->
        RecursiveCharacterTextSplitter, returns a list of chunk objects
        (each carrying its source document name and page number for later citation)
"""

from pathlib import Path


def load_documents(folder_path: str) -> list:
    """Load every PDF in folder_path and return a list of raw document objects."""
    raise NotImplementedError


def split_into_chunks(documents: list, chunk_size: int = 800, overlap: int = 100) -> list:
    """Split loaded documents into overlapping chunks, preserving source metadata."""
    raise NotImplementedError
