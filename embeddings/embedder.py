import numpy as np
from typing import List, Union

from embeddings.model import load_embedding_model


class TextEmbedder:
    def __init__(
        self,
        model_name: str =  "deepvk/USER-bge-m3",
        normalize: bool = True,
        device: str = None,
    ):
        self.model = load_embedding_model(
            model_name=model_name,
            normalize=normalize,
            device=device,
        )

    def embed_text(self, text: str) -> np.ndarray:
        vector = self.model.embed_query(text)
        return np.asarray(vector, dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.embed_documents(texts)
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(
        self,
        docs: List[Union[str, dict]],
        content_key: str = "page_content",
    ) -> np.ndarray:
        texts = []

        for doc in docs:
            if isinstance(doc, dict):
                texts.append(doc.get(content_key, ""))
            else:
                texts.append(doc.page_content)

        return self.embed_texts(texts)


if __name__ == "__main__":
    embedder = TextEmbedder(model_name="deepvk/USER-bge-m3")

    query_vec = embedder.embed_text("Как восстановить пароль?")
    print("Query vector:", query_vec.shape)

    docs = [
        {"page_content": "Инструкция по восстановлению пароля"},
        {"page_content": "Как изменить номер телефона"},
    ]

    doc_vecs = embedder.embed_documents(docs)
    print("Docs vectors:", doc_vecs.shape)
