FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY main.py .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# התקנה בסדר קריטי - אל תשנה!
RUN pip install --no-cache-dir "numpy>=2.0,<3.0"
RUN pip install --no-cache-dir torch==2.4.0 torchaudio==2.4.0
RUN pip install --no-cache-dir lightning==2.3.0
RUN pip install --no-cache-dir pyannote-audio==3.1.1
RUN pip install --no-cache-dir openai-whisper
RUN pip install --no-cache-dir runpod==1.6.2 requests soundfile

# בדיקות - אם נכשל כאן, הבנייה תיעצר
RUN python -c "import numpy as np; print('NumPy:', np.__version__); assert np.__version__.startswith('2'), 'Wrong NumPy!'"
RUN python -c "import pyannote.audio; print('pyannote.audio:', pyannote.audio.__version__); assert pyannote.audio.__version__.startswith('3'), 'Wrong pyannote!'"
RUN python -c "from pyannote.audio import Pipeline; print('✅ Pipeline OK')"

RUN python -c "import whisper; whisper.load_model('base')"

EXPOSE 8000
CMD ["python", "-u", "main.py"]
