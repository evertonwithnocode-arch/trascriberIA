FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Configurações básicas
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "runpod_serverless.py"]
