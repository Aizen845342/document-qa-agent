
"""Document Q&A Agent — Streamlit application."""

from pathlib import Path
import sys

import streamlit as st

# Make the existing backend modules importable.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingest import load_documents, split_into_chunks
from embeddings import build_vector_store, load_vector_store
from agent import build_qa_chain, answer_question

DOCUMENTS_DIR = ROOT / "documents"
INDEX_DIR = ROOT / "data" / "chroma"

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Document Q&A Agent",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Document Q&A Agent")
st.caption(
    "Ask questions about your PDF documents using "
    "retrieval-augmented generation and source citations."
)


def index_exists():
    """Check whether a populated Chroma index exists."""
    return INDEX_DIR.exists() and any(INDEX_DIR.iterdir())


@st.cache_resource(show_spinner="Loading document index...")
def get_vector_store():
    """Load the existing index, if available."""
    if index_exists():
        return load_vector_store(str(INDEX_DIR))
    return None


@st.cache_resource(show_spinner="Loading question-answering pipeline...")
def get_qa_chain():
    """Build the RAG chain using the existing backend."""
    store = get_vector_store()

    if store is None:
        return None

    return build_qa_chain(store)


def clear_cached_resources():
    """Refresh the vector store and RAG chain."""
    get_qa_chain.clear()
    get_vector_store.clear()


def index_documents():
    """Index all saved PDFs without deleting locked database files."""
    documents = load_documents(str(DOCUMENTS_DIR))

    if not documents:
        return 0, 0

    chunks = split_into_chunks(documents)

    if not chunks:
        return len(documents), 0

    # Release Streamlit's cached references before rebuilding.
    clear_cached_resources()

    # Rebuild the existing Chroma collection through its API.
    # This avoids deleting database files while Windows has them open.
    if index_exists():
        store = load_vector_store(str(INDEX_DIR))
        store.reset_collection()
        store.add_documents(chunks)
    else:
        build_vector_store(chunks, str(INDEX_DIR))

    clear_cached_resources()

    return len(documents), len(chunks)

# -----------------------------
# Sidebar: document management
# -----------------------------

with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button(
        "Save and index PDFs",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        try:
            for uploaded_file in uploaded_files:
                filename = Path(uploaded_file.name).name
                destination = DOCUMENTS_DIR / filename
                destination.write_bytes(uploaded_file.getvalue())

            with st.spinner("Processing documents and building embeddings..."):
                page_count, chunk_count = index_documents()

            if chunk_count:
                st.success(
                    f"Indexed {page_count} pages into "
                    f"{chunk_count} text chunks."
                )
            else:
                st.warning(
                    "No readable PDF text was found. "
                    "Scanned documents may require OCR."
                )

        except Exception as error:
            st.error(f"Document indexing failed: {error}")

    st.divider()

    saved_pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    st.subheader("Document library")
    st.write(f"**PDF files:** {len(saved_pdfs)}")

    if saved_pdfs:
        for pdf in saved_pdfs:
            st.caption(f"📄 {pdf.name}")
    else:
        st.caption("Upload a PDF to get started.")

    if st.button("Refresh document index", use_container_width=True):
        clear_cached_resources()
        st.rerun()

    st.divider()

    st.caption("Language model: Llama 3.1 8B")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Vector database: ChromaDB")


# -----------------------------
# Main interface
# -----------------------------

st.subheader("Ask your documents")

question = st.text_input(
    "Your question",
    placeholder="What does the document explain about...?",
)

ask_clicked = st.button(
    "Generate answer",
    type="primary",
    disabled=not question.strip(),
)

if ask_clicked:
    try:
        chain = get_qa_chain()

        if chain is None:
            st.warning(
                "Please upload and index at least one PDF "
                "before asking questions."
            )

        else:
            with st.spinner("Retrieving context and generating an answer..."):
                result = answer_question(chain, question.strip())

            st.divider()

            st.subheader("Answer")
            st.write(result.get("answer", "No answer was returned."))

            st.subheader("Sources")

            sources = result.get("sources", [])

            if sources:
                seen = set()

                for source in sources:
                    filename = source.get("source", "Unknown document")
                    page = source.get("page", "Unknown")

                    source_key = (filename, str(page))

                    if source_key in seen:
                        continue

                    seen.add(source_key)

                    st.markdown(
                        f"- **{filename}** — Page {page}"
                    )
            else:
                st.info("No source references were returned.")

    except Exception as error:
        st.error("Unable to generate an answer.")
        st.exception(error)
        st.info(
            "Check that Ollama is running and "
            "the llama3.1:8b model is installed."
        )
