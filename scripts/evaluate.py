import pickle
import numpy as np
from langchain_groq import ChatGroq 

from embeddings.embedder import TextEmbedder
from evaluation.llm_judge import LLMJudge
from evaluation.uniformity import uniformity_report


CHUNKS_PATH = "data/processed/chunks.pkl"


def main():
    print("Loading chunks...")

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"Chunks loaded: {len(chunks)}")

    embedder = TextEmbedder()

    print("Computing embeddings...")
    vectors = embedder.embed_documents(chunks)

    print("Uniformity:")
    report = uniformity_report(vectors)
    print(report)

    np.save("data/processed/embeddings.npy", vectors)
    print("Done")

def run_llm_judge(llm_name, temperature, max_tokens, dataset_to_check, model_answers):
    llm = ChatGroq(model=llm_name, temperature=temperature, max_tokens=max_tokens, timeout=None, max_retries=2)
    judge = LLMJudge(llm)

    report = judge.judge_batch(
        dataset=dataset_to_check,
        model_answers=model_answers,
    )

    print("\nMean Score:", report["mean_score"])

    return report


if __name__ == "__main__":
    main()
