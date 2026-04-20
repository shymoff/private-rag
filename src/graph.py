import os

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END

from src.state import AgentState

DB_PATH = "chroma_db/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 3
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(model="llama3:8b", temperature=0, base_url=OLLAMA_BASE_URL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})

GRADER_SYSTEM = (
    "Jesteś sędzią oceniającym trafność dokumentów. "
    "Odpowiedz tylko jednym słowem: 'tak' lub 'nie'."
)
grader_prompt = ChatPromptTemplate.from_messages([
    ("system", GRADER_SYSTEM),
    ("human", "Pytanie: {question}\n\nDokument: {document}"),
])
grader_chain = grader_prompt | llm

RAG_SYSTEM = (
    "Jesteś asystentem odpowiadającym na pytania na podstawie dostarczonego kontekstu. "
    "Jeśli w kontekście nie ma odpowiedzi, powiedz że nie wiesz. Odpowiadaj zwięźle."
)
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    ("human", "Kontekst:\n{context}\n\nPytanie: {question}"),
])
rag_chain = rag_prompt | llm


def retrieve_node(state: AgentState):
    """Pobiera najbardziej podobne fragmenty z chroma_db/."""
    print("--- RETRIEVAL ---")
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs]}


def grade_documents(state: AgentState):
    """Ocenia czy znalezione dokumenty są istotne dla pytania."""
    print("--- OCENA DOKUMENTÓW ---")

    documents = state["documents"]
    if not documents:
        return {"is_relevant": False}

    result = grader_chain.invoke({
        "question": state["question"],
        "document": documents[0],
    })
    return {"is_relevant": "tak" in result.content.lower()}


def generate(state: AgentState):
    """Generuje finalną odpowiedź na podstawie kontekstu."""
    print("--- GENEROWANIE ODPOWIEDZI ---")
    context = "\n\n".join(state["documents"])
    result = rag_chain.invoke({
        "context": context,
        "question": state["question"],
    })
    return {"generation": result.content}


def build_app():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges(
        "grade",
        lambda state: "generate" if state["is_relevant"] else "end",
        {"generate": "generate", "end": END},
    )
    workflow.add_edge("generate", END)

    return workflow.compile()


app = build_app()
