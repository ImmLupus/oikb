# ── Build ──
FROM python:3.12-slim AS builder

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir uv && \
    uv build --wheel

# ── Runtime ──
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/open-webui/oikb"
LABEL org.opencontainers.image.description="CLI tool for syncing content to Open WebUI Knowledge Bases"

# Install oikb from the built wheel
COPY --from=builder /app/dist/*.whl /tmp/
RUN whl=$(echo /tmp/*.whl) && \
    pip install --no-cache-dir "${whl}[qdrant]" && \
    rm /tmp/*.whl

# Optional Qdrant sparse BM25 for Confluence (enabled when QDRANT_URL is set).
ENV QDRANT_URL="http://sirhelper.komus.net:6333" \
    #QDRANT_API_KEY="" \
    QDRANT_BM25_COLLECTION=oikb-bm25

# Sync source is mounted at /data by convention.
VOLUME ["/data"]
WORKDIR /data

ENTRYPOINT ["oikb"]
