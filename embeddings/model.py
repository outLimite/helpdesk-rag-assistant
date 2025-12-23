from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional
import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embedding_model(
    model_name: str = "deepvk/USER-bge-m3",
    normalize: bool = True,
    device: Optional[str] = None,
):
    if device is None:
        device = get_device()

    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
    )

    return model
