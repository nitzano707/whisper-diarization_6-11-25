FROM python:3.10-slim

WORKDIR /app

# התקנת תלות מערכת
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# העתקת קוד
COPY main.py .

# שדרוג pip
RUN pip install --no-cache-dir --upgrade pip

# התקנת WhisperX ותלות
RUN pip install --no-cache-dir \
    git+https://github.com/m-bain/whisperx.git \
    runpod==1.6.2 \
    requests

# בדיקה
RUN python -c "import whisperx; print('✅ WhisperX installed')"

EXPOSE 8000

CMD ["python", "-u", "main.py"]
