from typing import List, Any
from vectorstore.faiss_store import FaissVectorStore
from embeddings.embedder import TextEmbedder


class Retriever:
    def __init__(self, db_path: str = None, model_name: str = None):
        self.store = FaissVectorStore(db_path, model_name=model_name)
        self.embedder = TextEmbedder(model_name=model_name)

    def retrieve(self, question: str, k: int = 5) -> List[Any]:
        return self.store.search(question, k=k)

    def retrieve_with_scores(self, question: str, k: int = 5):
        return self.store.search_with_scores(question, k=k)

    def retrieve_by_vector(self, vector, k: int = 5):
        if hasattr(self.store.db, "similarity_search_by_vector"):
            return self.store.db.similarity_search_by_vector(vector, k=k)
        return self.retrieve("", k=k)


if __name__=="__main__":
    retriever = Retriever(db_path="data/db/faiss_index", model_name="deepvk/USER-bge-m3")

    question = "Как отвязать VK ID от почты?"

    results = retriever.retrieve(question, k=5)
    print("Результаты поиска")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc}")

    results_with_scores = retriever.retrieve_with_scores(question, k=5)
    print("\nРезультаты с оценками")
    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"{i}. {doc} (score: {score})")

    vector = retriever.embedder.embed_text(question)
    vector_results = retriever.retrieve_by_vector(vector, k=5)
    print("\nПоиск по вектору")
    for i, doc in enumerate(vector_results, 1):
        print(f"{i}. {doc}")