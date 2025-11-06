import os
import runpod
import requests
import tempfile
import whisperx
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"Device: {device}")

def handler(event):
    try:
        input_data = event.get("input", {})
        file_url = input_data.get("file_url")
        language = input_data.get("language", "he")
        do_diarize = input_data.get("diarize", False)
        
        if not file_url:
            return {"error": "file_url required"}
        
        # הורדת קובץ
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()
            tmp.write(response.content)
            tmp.flush()
            audio_path = tmp.name
        
        print(f"Processing: {audio_path}")
        
        # תמלול עם WhisperX
        model = whisperx.load_model("base", device, compute_type=compute_type, language=language)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        
        transcription = " ".join([seg["text"] for seg in result["segments"]])
        
        speakers = []
        
        # דיאריזציה אם מבוקש
        if do_diarize:
            HF_TOKEN = os.getenv("HF_TOKEN")
            if not HF_TOKEN:
                return {"error": "HF_TOKEN required for diarization"}
            
            # Align
            model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
            # Diarize
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # חילוץ דוברים
            for seg in result["segments"]:
                if "speaker" in seg:
                    speakers.append({
                        "speaker": seg["speaker"],
                        "start": round(seg["start"], 2),
                        "end": round(seg["end"], 2),
                        "text": seg["text"]
                    })
        
        # ניקוי
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
    print("🚀 Starting WhisperX Runpod Worker")
    runpod.serverless.start({"handler": handler})
