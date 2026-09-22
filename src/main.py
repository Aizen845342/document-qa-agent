"""
CLI entry point: load the vector store (building it first if needed), then run
an interactive question loop against the RAG agent.

Run with: python src/main.py
"""

from pathlib import Path

from ingest import load_documents, split_into_chunks
from embeddings import build_vector_store, load_vector_store
from agent import build_qa_chain, answer_question

DOCUMENTS_DIR = "documents"
PERSIST_DIR = "data/chroma"


def get_or_build_vector_store():
    if Path(PERSIST_DIR).exists():
        return load_vector_store(PERSIST_DIR)

    documents = load_documents(DOCUMENTS_DIR)
    chunks = split_into_chunks(documents)
    return build_vector_store(chunks, PERSIST_DIR)


def main():
    vector_store = get_or_build_vector_store()
    chain = build_qa_chain(vector_store)

    print("Document Q&A Agent ready. Type a question (or 'exit' to quit).")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        result = answer_question(chain, question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {result['sources']}")


if __name__ == "__main__":
    main()
