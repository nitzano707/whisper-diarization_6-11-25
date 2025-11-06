FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY main.py .

RUN pip install --no-cache-dir --upgrade pip

# Faster Whisper - קל וזריז!
RUN pip install --no-cache-dir \
    faster-whisper \
    pyannote.audio==3.3.2 \
    runpod==1.6.2 \
    requests

RUN pip cache purge

EXPOSE 8000

CMD ["python", "-u", "main.py"]
