# Document Q&A Agent (RAG)

A retrieval-augmented question-answering agent: ask natural-language questions about a set of PDFs or technical documents and get grounded, source-cited answers instead of hallucinated ones.

**Status: in progress.** This repo currently holds the project scaffold and stub modules. Implementation is being built module by module, in the order below.

## Why this project

Built as hands-on practice with the core building blocks of production LLM applications: document ingestion, embeddings, vector retrieval, prompt-grounded generation, and — the part most tutorials skip — actually evaluating whether the system's answers are correct and honestly sourced, not just plausible-sounding.

## Pipeline overview

```
ingest PDFs --> chunk text --> embed chunks (OpenAI) --> store in Chroma
    --> [user question] --> embed question --> retrieve top-k chunks
    --> construct grounded prompt --> generate answer (OpenAI + LangChain)
    --> cite sources --> [evaluation set scores accuracy + catches hallucinations]
```

## Project structure

```
document-qa-agent/
├── documents/                # source PDFs to ingest (gitignored, user-supplied)
├── data/                     # persisted Chroma vector store (gitignored, generated locally)
├── eval/
│   └── qa_eval_set.json      # hand-written question/answer pairs for evaluation
├── src/
│   ├── ingest.py             # load PDFs, split into chunks
│   ├── embeddings.py         # embed chunks with OpenAI, build/persist the Chroma store
│   ├── retriever.py          # embed a query, retrieve top-k relevant chunks
│   ├── prompts.py            # prompt templates that force grounding + citation
│   ├── agent.py              # LangChain RAG chain tying retrieval + generation together
│   ├── evaluate.py           # run the eval set, score accuracy, flag hallucinations
│   └── main.py               # CLI entry point: ask a question, get a cited answer
├── app.py                    # optional Streamlit/Gradio UI wrapper
├── tests/                    # unit tests for each module
├── requirements.txt
├── .env.example              # OPENAI_API_KEY placeholder — never commit real keys
├── .gitignore
└── README.md
```

## Build order (and why)

| Step | Module | Depends on | Why this order |
|---|---|---|---|
| 1 | `ingest.py` | — | Nothing else works without loaded, chunked documents |
| 2 | `embeddings.py` | ingest | Chunks must exist before they can be embedded and stored |
| 3 | `retriever.py` | embeddings | Retrieval needs a populated vector store to search against |
| 4 | `prompts.py` | — (parallel) | Prompt design can be built alongside retrieval, wired in next |
| 5 | `agent.py` | retriever, prompts | The RAG chain combines retrieval output with the grounded prompt |
| 6 | `main.py` | agent | A usable CLI once the chain works end to end |
| 7 | `evaluate.py` | agent | Evaluation only makes sense once there's a working agent to evaluate |
| 8 | `app.py` | agent | UI wrapper is the last, optional polish layer |

## Methodology notes

- **Chunking strategy matters.** Chunks that are too large dilute retrieval relevance; too small loses context. Start with a fixed-size splitter (~500-1000 characters with overlap) and revisit if retrieval quality suffers.
- **Grounding is enforced in the prompt, not assumed.** The prompt explicitly instructs the model to answer only from the retrieved context and to say "I don't know" rather than guess when the context doesn't contain the answer.
- **Every answer cites its source chunk(s)/document(s).** This is what makes the system trustworthy rather than a black box — a user (or a reviewer) can verify the answer against the original text.
- **The evaluation set is the honesty check.** A hand-written set of question/answer pairs with known-correct answers, scored for: correct, partially correct, wrong, or hallucinated (confidently wrong with no basis in the retrieved context). Failures are diagnosed (bad retrieval vs. bad chunking vs. a prompt that doesn't constrain the model) and fixed before calling the project done.

## Getting started

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
python src/main.py
```

## Results

_To be filled in once the pipeline is built — will include retrieval quality notes, evaluation set scores, common failure modes found and fixed, and a short demo._
