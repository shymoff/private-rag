from collections import Counter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_PATH = "chroma_db/"
SAMPLE_SIZE = 3
SNIPPET_CHARS = 300


def run_inspection():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    data = vector_db.get()
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    print(f"--- Baza: {DB_PATH} ---")
    print(f"Łączna liczba fragmentów: {len(ids)}")

    if not ids:
        print("Baza jest pusta. Uruchom najpierw src/ingest.py.")
        return

    print("\n--- Fragmenty per plik źródłowy ---")
    sources = Counter(m.get("source", "?") for m in metadatas)
    for source, count in sources.most_common():
        print(f"  {count:4d}  {source}")

    print(f"\n--- Próbka pierwszych {SAMPLE_SIZE} fragmentów ---")
    for i in range(min(SAMPLE_SIZE, len(ids))):
        meta = metadatas[i] or {}
        snippet = documents[i][:SNIPPET_CHARS].replace("\n", " ")
        print(f"\n[{i}] id={ids[i]}")
        print(f"    source={meta.get('source', '?')}  page={meta.get('page', '?')}")
        print(f"    text={snippet}...")


if __name__ == "__main__":
    run_inspection()
