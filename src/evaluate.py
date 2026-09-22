"""
Step 7: Run a hand-written evaluation set against the agent and score its answers.

Pseudocode:
    load_eval_set(path) -> load question/expected_answer pairs from eval/qa_eval_set.json

    score_answer(question, expected_answer, actual_answer, sources) ->
        classify as one of: correct, partially_correct, wrong, hallucinated
        (hallucinated = confidently answered with no support in the cited sources)
        return the classification (and optionally a short reason)

    run_evaluation(chain, eval_set) ->
        for each (question, expected_answer) in eval_set:
            result = answer_question(chain, question)
            score = score_answer(question, expected_answer, result['answer'], result['sources'])
            record score
        print a summary: % correct, % partially correct, % wrong, % hallucinated
        return the full results table
"""

import json


def load_eval_set(path: str = "eval/qa_eval_set.json") -> list:
    """Load the hand-written question/expected_answer evaluation pairs."""
    with open(path) as f:
        return json.load(f)


def score_answer(question: str, expected_answer: str, actual_answer: str, sources: list) -> str:
    """Classify an answer as correct / partially_correct / wrong / hallucinated."""
    raise NotImplementedError


def run_evaluation(chain, eval_set: list) -> list:
    """Run the full evaluation set through the chain and return scored results."""
    raise NotImplementedError
