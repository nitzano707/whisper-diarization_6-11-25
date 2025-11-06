FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# התקנת Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# קישור python3 ל-python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY main.py .

# שדרוג pip
RUN pip3 install --no-cache-dir --upgrade pip

# התקנת חבילות
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    git+https://github.com/m-bain/whisperx.git \
    runpod==1.6.2 \
    requests

# בדיקה
RUN python -c "import whisperx; print('✅ OK')"

EXPOSE 8000

CMD ["python", "-u", "main.py"]
