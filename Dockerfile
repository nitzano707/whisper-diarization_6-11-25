FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY main.py .

RUN pip install --no-cache-dir --upgrade pip

# Whisper + Pyannote - פשוט ועובד!
RUN pip install --no-cache-dir \
    openai-whisper \
    pyannote.audio \
    runpod==1.6.2 \
    requests

RUN pip cache purge

# הורדת מודל Whisper מראש
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8000

CMD ["python", "-u", "main.py"]
