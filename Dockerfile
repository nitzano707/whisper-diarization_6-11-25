FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY main.py .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# כפיית NumPy 1.x
RUN pip install --no-cache-dir "numpy<2.0"

# PyTorch 2.0.x (תואם NumPy 1.x)
RUN pip install --no-cache-dir torch==2.0.1 torchaudio==2.0.2

# Lightning ישן
RUN pip install --no-cache-dir lightning==2.0.9

# pyannote-audio 2.1.1 (תואם NumPy 1.x)
RUN pip install --no-cache-dir pyannote-audio==2.1.1

# Whisper
RUN pip install --no-cache-dir openai-whisper

# Runpod + שאר
RUN pip install --no-cache-dir runpod==1.6.2 requests soundfile

# בדיקות
RUN python -c "import numpy as np; print('NumPy:', np.__version__); assert np.__version__.startswith('1'), 'Wrong NumPy!'"
RUN python -c "from pyannote.audio import Pipeline; print('✅ Pipeline OK')"

# הורדת Whisper model
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8000
CMD ["python", "-u", "main.py"]
