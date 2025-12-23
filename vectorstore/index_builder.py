import os
import sys
from typing import List, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from embeddings.embedder import TextEmbedder


def build_faiss_index(
    documents: List[Union[Document, dict]],
    save_path: str = "data/db/faiss_index",
    model_name: str = "deepvk/USER-bge-m3",
):
    os.makedirs(save_path, exist_ok=True)

    embedder = TextEmbedder(model_name=model_name)

    prepared_docs = []
    for doc in documents:
        if isinstance(doc, dict):
            prepared_docs.append(
                Document(
                    page_content=doc.get("page_content", ""),
                    metadata=doc.get("metadata", {}),
                )
            )
        else:
            prepared_docs.append(doc)

    print(f"Documents: {len(prepared_docs)}")

    db = FAISS.from_documents(
        prepared_docs,
        embedder.model,  
    )

    db.save_local(save_path)

    print(f"Index saved to: {save_path}")
    return db
