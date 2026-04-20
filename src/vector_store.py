from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Embeddingi są współdzielone na cały proces — model HF jest ciężki (~80MB RAM),
# nie ma sensu ładować go per sesja.
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_db() -> Chroma:
    """Tworzy nową, pustą, in-memory bazę wektorową — jedną per sesja czatu."""
    return Chroma(embedding_function=embeddings)
