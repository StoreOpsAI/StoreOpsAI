FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml M001_StoreOps_AI_매장점검규정_v0.1.md ./
COPY storeops_qna ./storeops_qna

EXPOSE 8002
CMD ["python", "-m", "storeops_qna.cli", "serve", "--host", "0.0.0.0", "--port", "8002"]
