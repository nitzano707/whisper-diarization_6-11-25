import os
import runpod
import requests
import tempfile
import whisper
from pyannote.audio import Pipeline

print("🚀 Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("✅ Whisper loaded")

print("🚀 Loading Diarization...")
HF_TOKEN = os.getenv("HF_TOKEN")
diarization_pipeline = None
if HF_TOKEN:
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        print("✅ Diarization loaded")
    except Exception as e:
        print(f"⚠️  Diarization failed: {e}")
else:
    print("⚠️  No HF_TOKEN - diarization disabled")

def handler(event):
    try:
        input_data = event.get("input", {})
        file_url = input_data.get("file_url")
        language = input_data.get("language", "he")
        do_diarize = input_data.get("diarize", True)
        
        if not file_url:
            return {"error": "file_url required"}
        
        print(f"📥 Downloading: {file_url}")
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()
            tmp.write(response.content)
            tmp.flush()
            audio_path = tmp.name
        
        print(f"✅ Downloaded: {os.path.getsize(audio_path)} bytes")
        
        # תמלול
        print("🎙️  Transcribing...")
        result = whisper_model.transcribe(audio_path, language=language, fp16=False)
        transcription = result["text"]
        print(f"✅ Transcribed: {len(transcription)} chars")
        
        # Segments בסיסיים
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip()
            })
        
        speakers = []
        
        # דיאריזציה
        if do_diarize and diarization_pipeline:
            try:
                print("👥 Diarizing...")
                diarization = diarization_pipeline(audio_path)
                
                # שיוך דוברים
                for segment in segments:
                    seg_start = segment["start"]
                    seg_end = segment["end"]
                    
                    # מצא דובר דומיננטי
                    speaker_times = {}
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        overlap_start = max(turn.start, seg_start)
                        overlap_end = min(turn.end, seg_end)
                        overlap = max(0, overlap_end - overlap_start)
                        
                        if overlap > 0:
                            speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap
                    
                    if speaker_times:
                        segment["speaker"] = max(speaker_times, key=speaker_times.get)
                    else:
                        segment["speaker"] = "SPEAKER_00"
                
                speakers = segments
                print(f"✅ Found {len(set([s.get('speaker', 'UNKNOWN') for s in speakers]))} speakers")
                
            except Exception as e:
                print(f"⚠️  Diarization failed: {e}")
                speakers = segments
        else:
            speakers = segments
        
        # ניקוי
        try:
            os.unlink(audio_path)
        except:
            pass
        
        return {
            "transcription": transcription,
            "speakers": speakers,
            "language": language,
            "status": "success"
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e), "status": "error"}

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Whisper + Diarization Worker")
    print("=" * 50)
    runpod.serverless.start({"handler": handler})
