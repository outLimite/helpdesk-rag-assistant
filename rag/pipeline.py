import os

from typing import Optional, Dict, Any
from rag.retriever import Retriever
from rag.generator import Generator


class RAGPipeline:
    def __init__(
            self,
            db_path: Optional[str] = "data/db/faiss_index",
            model_name: Optional[str] ="deepvk/USER-bge-m3",
            llm_name: Optional[str] = "llama-3.1-8b-instant"
    ):
        self.retriever = Retriever(db_path=db_path, model_name=model_name)
        self.generator = Generator(llm_name=llm_name)

    def answer(self, question: str, k: int = 5) -> Dict[str, Any]:
        docs = self.retriever.retrieve(question, k=k)

        sources = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            sources.append(
                {
                    "title": meta.get("title"),
                    "source": meta.get("source") or meta.get("url"),
                    "snippet": (getattr(d, "page_content", "") or "")[:600],
                }
            )

        answer = self.generator.generate(question, docs)

        return {"answer": answer, "sources": sources}

    def answer_with_scores(self, question: str, k: int = 5) -> Dict[str, Any]:
        docs_and_scores = self.retriever.retrieve_with_scores(question, k=k)
        docs = [d for d, s in docs_and_scores]
        scores = [s for d, s in docs_and_scores]

        sources = []
        for d, s in docs_and_scores:
            meta = getattr(d, "metadata", {}) or {}
            sources.append(
                {
                    "title": meta.get("title"),
                    "source": meta.get("source") or meta.get("url"),
                    "score": float(s),
                    "snippet": (getattr(d, "page_content", "") or "")[:600],
                }
            )

        answer = self.generator.generate(question, docs)
        return {"answer": answer, "sources": sources, "scores": scores}

if __name__=="__main__":

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"



    pipeline = RAGPipeline(
        db_path="data/db/faiss_index",
        model_name="deepvk/USER-bge-m3",
        llm_name="llama-3.1-8b-instant"
    )

    question = "Как отвязать VK ID от почты?"
    result = pipeline.answer(question, k=5)
    print("Ответ")
    print(result["answer"])
    print("\nИсточники")
    for src in result["sources"]:
        print(f"{src['title'] or 'No title'} ({src.get('source') or 'No source'})")
        print(f"{src['snippet']}\n")

    result_with_scores = pipeline.answer_with_scores(question, k=5)
    print("Ответ с оценками")
    print(result_with_scores["answer"])
    print("\nИсточники с оценками")
    for src in result_with_scores["sources"]:
        print(f"- {src['title'] or 'No title'} ({src.get('source') or 'No source'}) score: {src['score']}")
        print(f"  {src['snippet']}\n")