FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgl1 libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 기본은 CPU용. GPU용은 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX} \
	&& sed '/^torch[<=>]/d;/^torchvision[<=>]/d;s/^opencv-python$/opencv-python-headless/' requirements.txt > requirements-docker.txt \
	&& pip install --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]