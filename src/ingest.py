from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def ingest_pdf(file_path: str, vector_db: Chroma) -> int:
    """Wczytuje PDF, dzieli na fragmenty i dopisuje do przekazanej bazy.

    Zwraca liczbę dodanych fragmentów.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    chunks = _splitter.split_documents(documents)
    vector_db.add_documents(chunks)
    return len(chunks)
