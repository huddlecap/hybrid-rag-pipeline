FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); TextCrossEncoder(model_name='BAAI/bge-reranker-base')"

COPY api/ /app/api/
COPY src/ /app/src/
COPY scripts/ /app/scripts/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
