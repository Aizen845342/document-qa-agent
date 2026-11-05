

import json

EVALUATOR_SYSTEM_INSTRUCTIONS = """
You are an evaluator for a Retrieval-Augmented Generation (RAG) system.

Evaluate the generated answer using the question, expected answer,
actual answer, and retrieved document chunks.

Score the answer across four independent dimensions:

1. Correctness:
   Does the answer provide factually accurate information
   compared with the expected answer?

2. Completeness:
   Does the answer include the important information
   required to answer the question?

3. Groundedness:
   Are the claims in the answer supported by the retrieved chunks?
   Penalize unsupported claims or invented information.

4. Relevance:
   Does the answer directly address the question without
   unnecessary or unrelated information?

Use this scoring scale for each dimension:
1 = Very poor
2 = Significant problems
3 = Partially satisfactory
4 = Mostly satisfactory
5 = Excellent

Do not assume an answer is grounded simply because it is correct.
Evaluate each dimension independently.

Return valid JSON with these fields:
- correctness
- completeness
- groundedness
- relevance
- overall_score
- explanation

The overall_score must be the arithmetic average of the
four dimension scores.

Return JSON only. Do not include Markdown formatting.
"""
from langchain_ollama import ChatOllama
from pathlib import Path

MODEL = "llama3.1:8b"
def load_eval_set(path: str = "eval/qa_eval_set.json") -> list:
    print("Loading evaluation file from:", Path(path).resolve())

    with open(path) as f:
        data = json.load(f)

    print("Number of evaluation questions:", len(data))
    print("First question:", data[0]["question"])

    return data

def score_answer(question: str, expected_answer: str, actual_answer: str, retrieved_chunks: list) -> dict:
    """Evaluate an answer across four dimensions using an LLM judge."""
    evaluator = ChatOllama(model=MODEL)
    evaluate_input = {
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "retrieved_chunks": "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
        }
    assessment_prompt_eval = f"{EVALUATOR_SYSTEM_INSTRUCTIONS}\n\nInput:\n{json.dumps(evaluate_input, indent=2)}"
    eval_answer = evaluator.invoke(assessment_prompt_eval)
    return json.loads(eval_answer.content)


def run_evaluation(chain, eval_set: list) -> list:
    """Run the full evaluation set through the chain and return scored results."""
    results = []
    for item in eval_set:
        question = item["question"]
        expected_answer = item["expected_answer"]
        actual_answer_dict = chain.invoke(question)
        actual_answer = actual_answer_dict["answer"]
        retrieved_chunks = actual_answer_dict["chunks"]
        score = score_answer(question, expected_answer, actual_answer, retrieved_chunks)
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "score": score
        })
    return results
def calculate_average_score(results: list) -> float:
    """Calculate the average overall evaluation score."""
    if not results:
        return 0.0
    
    total_score = 0

    for result in results:
        total_score += result["score"]["overall_score"]
        
    average = total_score / len(results)

    return average