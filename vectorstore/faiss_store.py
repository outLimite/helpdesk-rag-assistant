import os
import sys
from typing import List
from langchain_community.vectorstores import FAISS

from embeddings.embedder import TextEmbedder


class FaissVectorStore:
    def __init__(
        self,
        db_path: str = "data/db/faiss_index",
        model_name: str = "deepvk/USER-bge-m3",
    ):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"FAISS index not found at: {db_path}")

        self.embedder = TextEmbedder(model_name=model_name)

        self.db = FAISS.load_local(
            db_path,
            self.embedder.model,
            allow_dangerous_deserialization=True,
        )

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List:
        results = self.db.similarity_search(query, k=k)
        return results

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
    ) -> List:
        results = self.db.similarity_search_with_score(query, k=k)
        return results

