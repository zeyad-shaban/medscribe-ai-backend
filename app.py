import io
import numpy as np
from scipy.io import wavfile
import librosa
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException
from utils.audio_cleaning import denoise_audio, normalize_audio
from config.settings import Settings, get_settings
from config import constants
from groq import Groq

groq_client = Groq(api_key=get_settings().groq_secret_key)
app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/transcribe")
async def process_and_transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    print(f"File size: {len(audio_bytes)} bytes")
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty file")

    buffer = io.BytesIO(audio_bytes)
    waveform, sr = librosa.load(buffer, sr=None)

    cleaned_audio = denoise_audio(waveform, sr)
    cleaned_audio = normalize_audio(cleaned_audio)

    # prepare audio to be sent
    audio_np = cleaned_audio.detach().cpu().numpy().squeeze()
    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=constants.GROQ_TARGET_SR)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    export_buffer = io.BytesIO()
    wavfile.write(export_buffer, constants.GROQ_TARGET_SR, audio_int16)
    export_buffer.seek(0)

    try:
        transcription = groq_client.audio.transcriptions.create(
            file=(file.filename, export_buffer.read()),
            model=constants.GROQ_MODEL_NAME,
            response_format="json",
            language="en",
        )
        return {
            "transcript": transcription.text,
            "filename": file.filename,
            "duration_seconds": round(len(waveform) / sr, 2),
        }

    except Exception as e:
        print(f"Groq API Error: {e}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("index:app", host="127.0.0.1", port=8000, reload=True)
