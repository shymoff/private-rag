# Private RAG

Lokalny chat z własnymi PDF-ami — w 100% offline. Wgrywasz dokumenty przez UI, zadajesz pytania, model odpowiada wyłącznie na podstawie ich zawartości. Nic nie wychodzi poza Twój komputer.

## Jak to działa

```
PDF → chunking → embeddingi (HuggingFace) → Chroma (in-memory per sesja)
                                                       ↓
                     pytanie → retrieval → grader (LLM) → generate (LLM) → odpowiedź
```

- **Embeddingi**: `sentence-transformers/all-MiniLM-L6-v2` (lokalnie przez HuggingFace)
- **LLM**: `llama3:8b` przez [Ollama](https://ollama.com/) (lokalnie)
- **Vector store**: ChromaDB, in-memory, per sesja przeglądarki
- **Orkiestracja**: LangGraph — retrieve → grade → generate, z warunkową krawędzią która zatrzymuje generację jeśli grader uzna dokumenty za nieistotne (brak halucynacji)
- **UI**: Streamlit z uploadem plików bezpośrednio w inpucie czatu

## Uruchomienie (Docker)

```bash
docker compose up -d --build
```

Pierwsza konfiguracja — pull modelu (jednorazowo):
```bash
docker compose exec ollama ollama pull llama3:8b
```

UI pod `http://localhost:8501`.

## Uruchomienie lokalne (bez Dockera)

Wymaga Pythona 3.13.2 i [uv](https://docs.astral.sh/uv/).

```bash
# 1. Zależności
uv sync

# 2. Ollama — w osobnym terminalu
ollama serve
ollama pull llama3:8b

# 3. Streamlit
uv run streamlit run app.py
```

## Użycie

1. Otwórz `http://localhost:8501`
2. Kliknij **+** przy polu czatu i wybierz PDF (możesz wgrać kilka naraz)
3. Poczekaj aż pojawi się "📎 Dodano **plik.pdf** — N fragmentów"
4. Zadaj pytanie

Jeśli grader uzna że dokumenty nie mają nic wspólnego z pytaniem, zobaczysz _"Nie znalazłem odpowiedzi w dokumentach."_ — to znaczy że RAG działa prawidłowo i model nie zmyśla.

## Izolacja per sesja

Każda zakładka przeglądarki ma **własną, odrębną bazę wektorową** w pamięci. Oznacza to:

- Dokumenty wgrane w zakładce A nie są widoczne w zakładce B
- Odświeżenie strony = czysty chat, trzeba wgrać pliki ponownie
- Żadne dane nie trafiają na dysk ani do zewnętrznych serwisów

## Struktura projektu

```
├── app.py                  # Streamlit UI
├── main.py                 # alternatywny CLI do grafu (sanity check)
├── Dockerfile
├── docker-compose.yml      # app + ollama + cache HuggingFace
├── pyproject.toml
└── src/
    ├── state.py            # AgentState (TypedDict)
    ├── vector_store.py     # fabryka create_vector_db() + shared embeddingi
    ├── ingest.py           # ingest_pdf(path, vector_db) — loader + chunking
    ├── graph.py            # build_app(vector_db) — LangGraph workflow
    └── inspect_db.py       # skrypt debugowy (nieużywany w nowym modelu)
```

## Zmienne środowiskowe

| Zmienna | Default | Opis |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Adres API Ollamy. W compose ustawiony na `http://ollama:11434` |

## Wydajność

- `llama3:8b` na CPU: **~30-60s na odpowiedź**
- Na GPU NVIDIA (po dodaniu sekcji `deploy.resources.reservations.devices` do serwisu `ollama` w compose): **~2-5s**
- Embeddingi liczą się raz przy wgraniu pliku, cache HuggingFace persystuje w wolumenie — drugie uruchomienie jest szybsze
