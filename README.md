# Private RAG

Chat with your own PDFs — 100% offline. Upload documents through the UI, ask questions, and the model answers strictly based on their contents. Nothing leaves your machine.

## How it works

```
PDF → chunking → embeddings (HuggingFace) → Chroma (in-memory, per session)
                                                       ↓
                    question → retrieval → grader (LLM) → generate (LLM) → answer
```

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, via HuggingFace)
- **LLM**: `llama3:8b` via [Ollama](https://ollama.com/) (local)
- **Vector store**: ChromaDB, in-memory, per browser session
- **Orchestration**: LangGraph — retrieve → grade → generate, with a conditional edge that skips generation if the grader marks the documents as irrelevant (no hallucinations)
- **UI**: Streamlit with file upload directly in the chat input

## Running with Docker

```bash
docker compose up -d --build
```

First-time setup — pull the model (one-off):
```bash
docker compose exec ollama ollama pull llama3:8b
```

UI at `http://localhost:8501`.

## Running locally (without Docker)

Requires Python 3.13.2 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Dependencies
uv sync

# 2. Ollama — in a separate terminal
ollama serve
ollama pull llama3:8b

# 3. Streamlit
uv run streamlit run app.py
```

## Usage

1. Open `http://localhost:8501`
2. Click **+** next to the chat input and pick a PDF (multiple files allowed)
3. Wait for the confirmation: "📎 Added **file.pdf** — N chunks"
4. Ask a question

If the grader decides the documents have nothing to do with the question, you'll see _"Nie znalazłem odpowiedzi w dokumentach."_ — this means RAG is working correctly and the model is not making things up.

## Per-session isolation

Every browser tab has its **own, isolated vector store** in memory. This means:

- Documents uploaded in tab A are not visible in tab B
- Refreshing the page = clean chat, files need to be re-uploaded
- No data is written to disk or sent to any external service

## Project structure

```
├── app.py                  # Streamlit UI
├── main.py                 # alternative CLI entry point (sanity check)
├── Dockerfile
├── docker-compose.yml      # app + ollama + HuggingFace cache
├── pyproject.toml
└── src/
    ├── state.py            # AgentState (TypedDict)
    ├── vector_store.py     # create_vector_db() factory + shared embeddings
    ├── ingest.py           # ingest_pdf(path, vector_db) — loader + chunking
    ├── graph.py            # build_app(vector_db) — LangGraph workflow
    └── inspect_db.py       # debug script (unused in the current model)
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API address. Set to `http://ollama:11434` in compose |

## Performance

- `llama3:8b` on CPU: **~30-60s per answer**
- On an NVIDIA GPU (after adding a `deploy.resources.reservations.devices` section to the `ollama` service in compose): **~2-5s**
- Embeddings are computed once per uploaded file; the HuggingFace cache persists in a volume, so subsequent runs are faster
