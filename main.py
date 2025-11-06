import os
import runpod
import requests
import tempfile
import whisper
from pyannote.audio import Pipeline

import numpy as np
import torch
import pyannote.audio
print(f"NumPy: {np.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"pyannote.audio: {pyannote.audio.__version__}")

print("Loading Whisper model...")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
whisper_model = whisper.load_model(WHISPER_MODEL)
print(f"✓ Whisper '{WHISPER_MODEL}' loaded")

print("Loading Diarization pipeline...")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        print("✓ Diarization pipeline loaded")
    except Exception as e:
        print(f"⚠ Diarization load failed: {e}")
        diarization_pipeline = None
else:
    print("⚠ HF_TOKEN not set")
    diarization_pipeline = None

def handler(event):
    try:
        input_data = event.get("input", {})
        file_url = input_data.get("file_url")
        language = input_data.get("language", "he")
        do_diarize = input_data.get("diarize", False)
        
        if not file_url:
            return {"error": "file_url is required"}
        
        print(f"Processing: {file_url}")
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()
            tmp.write(response.content)
            tmp.flush()
            audio_path = tmp.name
        
        print(f"Downloaded: {os.path.getsize(audio_path)} bytes")
        
        print("Transcribing...")
        result = whisper_model.transcribe(audio_path, language=language, fp16=False)
        transcription = result["text"]
        print(f"✓ Transcription: {len(transcription)} chars")
        
        speakers = []
        if do_diarize:
            if not diarization_pipeline:
                return {"error": "Diarization not available"}
            
            print("Diarizing...")
            diarization = diarization_pipeline(audio_path)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({
                    "speaker": speaker,
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2)
                })
            print(f"✓ Found {len(speakers)} segments")
        
        try:
            os.unlink(audio_path)
        except:
            pass
        
        return {
            "transcription": transcription,
            "speakers": speakers,
            "language": language
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Runpod Serverless Worker Starting")
    print("=" * 50)
    runpod.serverless.start({"handler": handler})
