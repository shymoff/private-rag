import streamlit as st

from src.graph import app as rag_app

st.set_page_config(page_title="Private RAG", page_icon="📄")
st.title("📄 Private RAG")


@st.cache_resource
def get_app():
    return rag_app


if "messages" not in st.session_state:
    st.session_state.messages = []

# Komunikat powitalny — widoczny dopóki użytkownik nic nie wyśle
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("Zadaj pytanie o zaindeksowane dokumenty.")

# Historia rozmowy
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input na dole
if prompt := st.chat_input("Twoje pytanie..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Myślę..."):
            result = get_app().invoke({
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
