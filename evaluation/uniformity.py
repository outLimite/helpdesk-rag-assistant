import torch
import numpy as np
from typing import Union


def _to_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return x.float()


def compute_uniformity(
    embeddings: Union[np.ndarray, torch.Tensor],
    t: float = 2.0,
) -> float:
    x = _to_tensor(embeddings)

    sq_pdist = torch.pdist(x, p=2).pow(2)
    uniformity = torch.log(torch.mean(torch.exp(-t * sq_pdist)))

    return uniformity.item()


def uniformity_report(embeddings: Union[np.ndarray, torch.Tensor]) -> dict:
    value = compute_uniformity(embeddings)
    return {
        "uniformity": round(value, 6),
        "vectors": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
    }

if __name__ == "__main__":
    x = np.random.randn(100, 128)
    print(uniformity_report(x))
