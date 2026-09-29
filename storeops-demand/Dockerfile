FROM python:3.12-slim

WORKDIR /app

COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY . .

ENV STOREOPS_DEMO=1
EXPOSE 8765

CMD ["python", "-m", "uvicorn", "storeops.api:app", "--host", "0.0.0.0", "--port", "8765"]