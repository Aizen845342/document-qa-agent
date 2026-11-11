# Document Q&A Agent (RAG)

A retrieval-augmented question-answering agent: ask natural-language questions about a set of PDFs or technical documents and get grounded, source-cited answers instead of relying on unsupported model-generated responses.

The project implements an end-to-end RAG pipeline using LangChain, Hugging Face embeddings, ChromaDB, and a locally hosted Llama 3.1 model through Ollama. A Streamlit interface supports document uploads, indexing, and interactive question answering.

## Why this project

Built as hands-on practice with the core building blocks of production LLM applications: document ingestion, embeddings, vector retrieval, prompt-grounded generation, and evaluating whether the system's answers are correct and supported by their sources, not just plausible-sounding.

The focus is on building a modular, locally runnable document QA system with transparent retrieval and source attribution.

## Pipeline overview

```text
ingest PDFs --> chunk text --> embed chunks (Hugging Face) --> store in Chroma
    --> [user question] --> embed question --> retrieve top-k chunks
    --> construct grounded prompt --> generate answer (Ollama + LangChain)
    --> cite sources --> [evaluation set scores answer quality]
```

**Technology stack:** Python, LangChain, Ollama (Llama 3.1 8B), Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`), ChromaDB, Streamlit, and Pytest.

The language model runs locally through Ollama, so an OpenAI API key is not required.

## Project structure

```text
document-qa-agent/
├── documents/                # source PDFs to ingest (gitignored, user-supplied)
├── data/                     # persisted Chroma vector store (gitignored, generated locally)
├── eval/
│   └── qa_eval_set.json      # question/answer pairs for evaluation
├── src/
│   ├── ingest.py             # load PDFs, split into overlapping chunks
│   ├── embeddings.py         # Hugging Face embeddings, build/persist the Chroma store
│   ├── retriever.py          # retrieve top-k relevant chunks using semantic similarity
│   ├── prompts.py            # grounded prompt construction and citation instructions
│   ├── agent.py              # LangChain RAG chain tying retrieval + generation together
│   ├── evaluate.py           # evaluate answer correctness and grounding
│   └── main.py               # CLI entry point for document question answering
├── app.py                    # Streamlit UI for PDF upload, indexing, and Q&A
├── tests/                    # project tests
├── requirements.txt
├── .gitignore
└── README.md
```

## Build order (and why)

| Step | Module | Depends on | Why this order |
|---|---|---|---|
| 1 | `ingest.py` | — | Documents must be loaded and chunked before retrieval can work |
| 2 | `embeddings.py` | ingest | Chunks must exist before they can be embedded and stored |
| 3 | `retriever.py` | embeddings | Retrieval requires a populated vector store |
| 4 | `prompts.py` | — (parallel) | Prompt construction defines how retrieved context is used |
| 5 | `agent.py` | retriever, prompts | The RAG chain combines retrieval with grounded generation |
| 6 | `main.py` | agent | Provides a command-line entry point |
| 7 | `evaluate.py` | agent | Measures answer quality using an evaluation dataset |
| 8 | `app.py` | agent | Provides the interactive document upload and Q&A interface |

## Methodology notes

- **Chunking strategy matters.** Chunks that are too large dilute retrieval relevance; chunks that are too small lose context. The system uses `RecursiveCharacterTextSplitter` with a chunk size of 800 characters and an overlap of 100 characters.
- **Semantic retrieval is more than keyword matching.** Document chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in ChromaDB. Questions are embedded using the same model, allowing retrieval based on semantic similarity.
- **Grounding is enforced in the prompt, not assumed.** The prompt instructs Llama 3.1 to answer only from the retrieved context and to say "I don't know" when the information is insufficient. Prompt instructions reduce unsupported answers but cannot guarantee their elimination.
- **Answers include source references.** The agent returns the document filenames and page metadata associated with retrieved chunks, allowing users to check the original material.
- **The evaluation set is the quality check.** A hand-written set of question/answer pairs is used to assess generated responses. Evaluation helps identify incorrect answers, missing information, weak retrieval, and unsupported claims.
- **Local inference keeps the workflow self-contained.** Ollama runs Llama 3.1 8B locally, while Hugging Face generates embeddings and ChromaDB persists the vector index.

## Evaluation

The project includes an evaluation module and a question/answer dataset under eval/.

The evaluation workflow compares generated responses against expected answers and assesses answer quality using an LLM-based evaluator.

Key areas of interest include:

Correctness: Whether the response matches the expected information.

Completeness: Whether the answer covers the important details.

Groundedness: Whether the response is supported by retrieved context.

Relevance: Whether the response addresses the original question.

Evaluation scores should be interpreted alongside manual inspection of retrieved passages and source references.

## Results

The application supports PDF ingestion, chunking, local embedding generation, persistent vector storage, semantic retrieval, grounded answer generation, and source attribution through a Streamlit interface.

The evaluation module provides a framework for measuring answer quality. Quantitative evaluation results and retrieval performance measurements will be reported after running the evaluation dataset.

## Future improvements

Potential extensions include hybrid search, reranking retrieved chunks, support for additional document formats, and more detailed retrieval evaluation.