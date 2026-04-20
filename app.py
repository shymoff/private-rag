import tempfile
from pathlib import Path

import streamlit as st

from src.graph import build_app
from src.ingest import ingest_pdf
from src.vector_store import create_vector_db

st.set_page_config(page_title="Private RAG", page_icon="📄")
st.title("📄 Private RAG")

# Per-sesja: własna baza wektorowa i własny graf RAG
if "vector_db" not in st.session_state:
    st.session_state.vector_db = create_vector_db()
    st.session_state.rag_app = build_app(st.session_state.vector_db)
    st.session_state.ingested = {}
    st.session_state.messages = []

# Komunikat powitalny — widoczny dopóki użytkownik nic nie wyśle
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Dodaj PDF przyciskiem **+** i zadaj pytanie o jego zawartość.")

# Historia rozmowy
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input na dole z przyciskiem dołączania plików
user_input = st.chat_input(
    "Twoje pytanie...",
    accept_file="multiple",
    file_type=["pdf"],
)

if user_input:
    # Najpierw: indeksuj wgrane pliki (jeśli są)
    for f in user_input.files or []:
        key = f"{f.name}-{f.size}"
        if key in st.session_state.ingested:
            continue
        with st.chat_message("assistant"):
            with st.spinner(f"Indeksuję {f.name}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name
                try:
                    n = ingest_pdf(tmp_path, st.session_state.vector_db)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                st.session_state.ingested[key] = (f.name, n)
            msg = f"📎 Dodano **{f.name}** — {n} fragmentów"
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})

    # Potem: jeśli było pytanie, odpowiedz
    if user_input.text:
        prompt = user_input.text
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Myślę..."):
                result = st.session_state.rag_app.invoke({
                    "question": prompt,
                    "documents": [],
                    "generation": "",
                    "is_relevant": False,
                })
                answer = result.get("generation") or (
                    "Nie znalazłem odpowiedzi w dokumentach."
                )
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
