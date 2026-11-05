from pathlib import Path

from ingest import load_documents, split_into_chunks
from embeddings import build_vector_store, load_vector_store
from agent import build_qa_chain, answer_question
from evaluate import (
    load_eval_set,
    run_evaluation,
    calculate_average_score
)

DOCUMENTS_DIR = "documents"
PERSIST_DIR = "data/chroma"


def get_or_build_vector_store():
    if Path(PERSIST_DIR).exists():
        return load_vector_store(PERSIST_DIR)

    documents = load_documents(DOCUMENTS_DIR)
    chunks = split_into_chunks(documents)

    return build_vector_store(chunks, PERSIST_DIR)

def display_evaluation_results(results):
    print("\n========== EVALUATION RESULTS ==========")

    for index, result in enumerate(results, start=1):
        score = result["score"]

        print(f"\nQuestion {index}: {result['question']}")
        print(f"Expected answer: {result['expected_answer']}")
        print(f"Generated answer: {result['actual_answer']}")

        print("\nScores:")
        print(f"Correctness:  {score['correctness']}/5")
        print(f"Completeness: {score['completeness']}/5")
        print(f"Groundedness: {score['groundedness']}/5")
        print(f"Relevance:    {score['relevance']}/5")
        print(f"Overall:      {score['overall_score']}/5")
        print(f"Explanation:  {score['explanation']}")

    average = calculate_average_score(results)

    print("\n========== SUMMARY ==========")
    print(f"Average overall score: {average:.2f}/5")

def main():
    vector_store = get_or_build_vector_store()
    chain = build_qa_chain(vector_store)

    print("Document Q&A Agent ready.")
    print("Ask a question, type 'evaluate' to test the RAG system, or 'exit' to quit.")

    while True:
        question = input("\n> ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Exiting Document Q&A Agent.")
            break

        if not question:
            continue

        if question.lower() == "evaluate":
            eval_set = load_eval_set()
            results = run_evaluation(chain, eval_set)
            display_evaluation_results(results)
            continue

        result = answer_question(chain, vector_store, question)

        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {result['sources']}")

        


if __name__ == "__main__":
    main()