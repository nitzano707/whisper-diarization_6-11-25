FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY main.py .

RUN pip install --no-cache-dir --upgrade pip

# NumPy 1.x
RUN pip install --no-cache-dir "numpy<2.0"

# PyTorch 2.0
RUN pip install --no-cache-dir torch==2.0.1 torchaudio==2.0.2

# pyannote-audio 2.1.1
RUN pip install --no-cache-dir \
    pyannote-audio==2.1.1 \
    pytorch-lightning==2.0.9

# Whisper
RUN pip install --no-cache-dir openai-whisper

# Runpod
RUN pip install --no-cache-dir runpod==1.6.2 requests soundfile

# וידוא
RUN python -c "import numpy; print('NumPy:', numpy.__version__)"
RUN python -c "from pyannote.audio import Pipeline; print('✅ Works!')"

# מודל Whisper
RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8000
CMD ["python", "-u", "main.py"]
