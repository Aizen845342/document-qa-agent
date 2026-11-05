
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from embeddings import load_vector_store
from retriever import retrieve_relevant_chunks
from prompts import build_qa_prompt
from pathlib import Path

MODEL = "llama3.1:8b"


def build_qa_chain(vector_store):
    """Construct and return the end-to-end LangChain RAG chain."""
    model = ChatOllama(
            model= MODEL        
    )
    def retrieve_chunks(question: str) -> dict:
        chunks = retrieve_relevant_chunks(vector_store, question, k=4)
        inputs = { 
            "question": question,
            "chunks": chunks
        }
        return inputs
    def build_prompt(inputs) -> str:
        question = inputs["question"]
        chunks = inputs["chunks"]
        prompt = build_qa_prompt(question, chunks) 
        return prompt
    chain = RunnableLambda(retrieve_chunks) |RunnablePassthrough.assign(answer=(RunnableLambda(build_prompt)) | model |StrOutputParser())
    return chain


def answer_question(chain,question: str) -> dict:
    """Run the chain on a question; return the answer plus cited sources."""
    final_answer = chain.invoke(question)
    sources = [{"source": Path(chunk.metadata["source"]).name, "page": chunk.metadata.get("page_label", chunk.metadata.get("page", "unknown"))} for chunk in final_answer["chunks"]]
    final_answer = {
        "question":final_answer["question"],
        "answer": final_answer["answer"],
        "sources": sources
    }
    return final_answer

if __name__ == "__main__":
    vector_store = load_vector_store()
    chain = build_qa_chain(vector_store)

    user_question = "What is the relationship between frequency and wavelength?"

    result = answer_question(chain, vector_store, user_question)

    print("Answer:", result["answer"])
    print("Sources:", result["sources"])