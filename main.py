import os
import runpod
import requests
import tempfile
import whisper
from pyannote.audio import Pipeline

print("Loading Whisper...")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
whisper_model = whisper.load_model(WHISPER_MODEL)
print(f"✓ Whisper loaded")

print("Loading Diarization...")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        print("✓ Diarization loaded")
    except Exception as e:
        print(f"⚠ Diarization failed: {e}")
        diarization_pipeline = None
else:
    print("⚠ No HF_TOKEN")
    diarization_pipeline = None

def handler(event):
    try:
        input_data = event.get("input", {})
        file_url = input_data.get("file_url")
        language = input_data.get("language", "he")
        do_diarize = input_data.get("diarize", False)
        
        if not file_url:
            return {"error": "file_url required"}
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()
            tmp.write(response.content)
            tmp.flush()
            audio_path = tmp.name
        
        result = whisper_model.transcribe(audio_path, language=language, fp16=False)
        transcription = result["text"]
        
        speakers = []
        if do_diarize and diarization_pipeline:
            diarization = diarization_pipeline(audio_path)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.append({
                    "speaker": speaker,
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2)
                })
        
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
    print("🚀 Starting Runpod Worker")
    runpod.serverless.start({"handler": handler})
