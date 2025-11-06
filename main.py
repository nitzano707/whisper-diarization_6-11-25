import os
import runpod
import requests
import tempfile
import whisperx
import torch

# הגדרות
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"🖥️  Device: {device}")
print(f"🔢 Compute type: {compute_type}")

def handler(event):
    """
    Handler for Runpod Serverless
    
    Input:
    {
        "input": {
            "file_url": "https://example.com/audio.mp3",
            "language": "he",  # optional
            "diarize": true    # optional
        }
    }
    """
    try:
        input_data = event.get("input", {})
        file_url = input_data.get("file_url")
        language = input_data.get("language", "he")
        do_diarize = input_data.get("diarize", True)
        
        if not file_url:
            return {"error": "file_url is required"}
        
        print(f"📥 Downloading: {file_url}")
        
        # הורדת קובץ אודיו
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()
            tmp.write(response.content)
            tmp.flush()
            audio_path = tmp.name
        
        print(f"✅ Downloaded: {os.path.getsize(audio_path)} bytes")
        
        # שלב 1: תמלול עם Whisper
        print("🎙️  Transcribing...")
        model = whisperx.load_model("base", device, compute_type=compute_type, language=language)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        
        # טקסט מלא
        transcription = " ".join([seg["text"] for seg in result["segments"]])
        print(f"✅ Transcribed: {len(transcription)} chars")
        
        speakers = []
        
        # שלב 2: דיאריזציה (אם מבוקש)
        if do_diarize:
            HF_TOKEN = os.getenv("HF_TOKEN")
            if not HF_TOKEN:
                print("⚠️  HF_TOKEN not set, skipping diarization")
            else:
                try:
                    print("🔍 Aligning...")
                    # Align למילים מדויקות
                    model_a, metadata = whisperx.load_align_model(
                        language_code=language, 
                        device=device
                    )
                    result = whisperx.align(
                        result["segments"], 
                        model_a, 
                        metadata, 
                        audio, 
                        device, 
                        return_char_alignments=False
                    )
                    
                    print("👥 Diarizing...")
                    # Diarization
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=HF_TOKEN, 
                        device=device
                    )
                    diarize_segments = diarize_model(audio)
                    
                    # שיוך דוברים למילים
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                    # חילוץ segments עם דוברים
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
                    # במקרה של כשלון, נחזיר segments ללא שיוך דוברים
                    for seg in result["segments"]:
                        speakers.append({
                            "speaker": "SPEAKER_00",
                            "start": round(seg["start"], 2),
                            "end": round(seg["end"], 2),
                            "text": seg["text"].strip()
                        })
        
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
            "details": error_details,
            "status": "error"
        }

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 WhisperX + Diarization - Runpod Serverless Worker")
    print("=" * 60)
    runpod.serverless.start({"handler": handler})
