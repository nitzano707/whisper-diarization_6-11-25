import os
import runpod
import requests
import tempfile
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

print("🚀 Loading models...")

# Faster Whisper - קל וזריז
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# Pyannote
HF_TOKEN = os.getenv("HF_TOKEN")
diarization_pipeline = None
if HF_TOKEN:
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        print("✅ Diarization loaded")
    except:
        print("⚠️  Diarization failed to load")

print("✅ Models ready")

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
        
        # תמלול עם Faster Whisper
        print("🎙️  Transcribing...")
        segments, info = whisper_model.transcribe(
            audio_path, 
            language=language,
            beam_size=5
        )
        
        # איסוף segments
        all_segments = []
        transcription_parts = []
        
        for segment in segments:
            all_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip()
            })
            transcription_parts.append(segment.text)
        
        transcription = " ".join(transcription_parts)
        print(f"✅ Transcribed: {len(transcription)} chars")
        
        speakers = []
        
        # דיאריזציה
        if do_diarize and diarization_pipeline:
            try:
                print("👥 Diarizing...")
                diarization = diarization_pipeline(audio_path)
                
                # שיוך דוברים ל-segments
                for segment in all_segments:
                    # מצא את הדובר הדומיננטי בזמן הזה
                    seg_start = segment["start"]
                    seg_end = segment["end"]
                    seg_duration = seg_end - seg_start
                    
                    speaker_times = {}
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        overlap_start = max(turn.start, seg_start)
                        overlap_end = min(turn.end, seg_end)
                        overlap = max(0, overlap_end - overlap_start)
                        
                        if overlap > 0:
                            speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap
                    
                    # בחר דובר עם הכי הרבה זמן
                    if speaker_times:
                        dominant_speaker = max(speaker_times, key=speaker_times.get)
                        segment["speaker"] = dominant_speaker
                    else:
                        segment["speaker"] = "SPEAKER_00"
                
                speakers = all_segments
                print(f"✅ Assigned speakers")
                
            except Exception as e:
                print(f"⚠️  Diarization failed: {e}")
                speakers = all_segments
        else:
            speakers = all_segments
        
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
    print("🚀 Faster-Whisper Worker Starting")
    runpod.serverless.start({"handler": handler})
