import os
import runpod
import requests
import tempfile
import whisperx
import torch
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"🖥️  Device: {device}")

def handler(event):
    try:
        input_data = event.get("input", {})
        file_url = input_data.get("file_url")
        language = input_data.get("language", "he")
        do_diarize = input_data.get("diarize", True)
        
        if not file_url:
            return {"error": "file_url required"}
        
        print(f"📥 Downloading: {file_url}")
        
        # הורדה
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()
            tmp.write(response.content)
            tmp.flush()
            audio_path = tmp.name
        
        print(f"✅ Downloaded: {os.path.getsize(audio_path)} bytes")
        
        # תמלול
        print("🎙️  Transcribing...")
        model = whisperx.load_model("base", device, compute_type=compute_type, language=language)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        
        transcription = " ".join([seg["text"] for seg in result["segments"]])
        print(f"✅ Transcribed: {len(transcription)} chars")
        
        # שחרור זיכרון
        del model
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None
        
        speakers = []
        
        # דיאריזציה
        if do_diarize:
            HF_TOKEN = os.getenv("HF_TOKEN")
            if HF_TOKEN:
                try:
                    print("🔍 Aligning...")
                    model_a, metadata = whisperx.load_align_model(
                        language_code=language, 
                        device=device
                    )
                    result = whisperx.align(
                        result["segments"], 
                        model_a, 
                        metadata, 
                        audio, 
                        device
                    )
                    
                    # שחרור זיכרון
                    del model_a
                    gc.collect()
                    torch.cuda.empty_cache() if device == "cuda" else None
                    
                    print("👥 Diarizing...")
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=HF_TOKEN, 
                        device=device
                    )
                    diarize_segments = diarize_model(audio)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                    for seg in result["segments"]:
                        speakers.append({
                            "speaker": seg.get("speaker", "UNKNOWN"),
                            "start": round(seg["start"], 2),
                            "end": round(seg["end"], 2),
                            "text": seg["text"].strip()
                        })
                    
                    print(f"✅ Found {len(set([s['speaker'] for s in speakers]))} speakers")
                    
                except Exception as e:
                    print(f"⚠️  Diarization failed: {e}")
        
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
        error_details = traceback.format_exc()
        print(f"❌ Error:\n{error_details}")
        return {
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    print("🚀 WhisperX Worker Starting")
    runpod.serverless.start({"handler": handler})
```

---

## ⚙️ **הגדרות Runpod - חשוב מאוד!**

בעת יצירת Endpoint, הגדר:
```
Container Disk: 20 GB
GPU: RTX 4090 או A40
Execution Timeout: 300 (5 דקות)
Max Workers: 3
Min Workers: 0
