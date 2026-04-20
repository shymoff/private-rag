FROM python:3.13.2-slim

WORKDIR /app

# uv do zarządzania zależnościami
RUN pip install --no-cache-dir uv

# Instalacja zależności (osobna warstwa dla cache)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Kod aplikacji
COPY . .

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
