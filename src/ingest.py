import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Konfiguracja ścieżek
DATA_PATH = "data/"
DB_PATH = "chroma_db/"

def run_ingestion():
    # 2. Wczytywanie dokumentów
    print("--- Wczytywanie PDF-ów ---")
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    # 3. Chunking (Dzielenie tekstu)
    # Używamy Recursive, bo próbuje dzielić tekst w logicznych miejscach (kropki, nowe linie)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200 # Zakładka, żeby nie tracić kontekstu między kawałkami
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Utworzono {len(chunks)} fragmentów tekstu.")

    # 4. Embeddingi (Lokalny model z Hugging Face)
    # Wybieramy all-MiniLM-L6-v2 - jest mały, szybki i świetny do testów
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5. Zapis do bazy wektorowej ChromaDB
    print("--- Indeksowanie w bazie wektorowej ---")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("✅ Baza gotowa i zapisana lokalnie!")

if __name__ == "__main__":
    run_ingestion()