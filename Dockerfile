FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY main.py .

RUN pip install --no-cache-dir --upgrade pip

# WhisperX - כולל diarization מובנה!
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git

# Runpod
RUN pip install --no-cache-dir runpod==1.6.2 requests

# בדיקה
RUN python -c "import whisperx; print('✅ WhisperX OK')"

EXPOSE 8000
CMD ["python", "-u", "main.py"]
