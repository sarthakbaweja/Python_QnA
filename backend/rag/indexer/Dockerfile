FROM python:3.11-slim

WORKDIR /app

COPY rag/indexer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en')"

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "-m", "rag.indexer.index_dataset"]
