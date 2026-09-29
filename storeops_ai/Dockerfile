FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
	&& sed '/^torch[<=>]/d;/^torchvision[<=>]/d;s/^opencv-python$/opencv-python-headless/' requirements.txt > requirements-docker.txt \
	&& pip install --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]