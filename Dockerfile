FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Instalar deps do sistema
RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Runpod entrypoint
CMD ["python3", "runpod_serverless.py"]
