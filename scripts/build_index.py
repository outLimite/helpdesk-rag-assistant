import pickle
from vectorstore.index_builder import build_faiss_index


CHUNKS_PATH = "data/processed/chunks.pkl"
FAISS_PATH = "data/db/faiss_index"


def main():
    print("Loading chunks...")

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"Loaded chunks: {len(chunks)}")

    print("Building FAISS index...")
    build_faiss_index(
        documents=chunks,
        save_path=FAISS_PATH,
    )

    print("FAISS index ready")


if __name__ == "__main__":
    main()
