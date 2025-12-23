import os
import sys
import pickle

from ingestion.loader import load_documents_from_site
from ingestion.cleaner import bs4_extractor
from ingestion.splitter import split_documents


RAW_PATH = "data/raw/raw_docs.pkl"
CHUNKS_PATH = "data/processed/chunks.pkl"

def save_documents(docs, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(docs, f)

def load_documents(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    print("Loading documents from site...")

    raw_docs = load_documents_from_site(
        "https://help.mail.ru/",
        max_depth=3,
        extractor=bs4_extractor,
        continue_on_failure=True,
    )

    save_documents(raw_docs, RAW_PATH)
    print(f"Raw documents saved: {len(raw_docs )}")

    print("Splitting into chunks...")
    chunks = split_documents(raw_docs)

    save_documents(raw_docs, RAW_PATH)
    print(f"Chunks saved: {len(chunks)}")


if __name__ == "__main__":
    main()
