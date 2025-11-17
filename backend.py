from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
import tempfile
import os

app = FastAPI()

# On garde tiny.en mais on le charge depuis le hub Hugging Face (pas dans le bundle)
model = WhisperModel("Systran/faster-whisper-tiny.en", download_root="/tmp", device="cpu", compute_type="int8")

app.mount("/", StaticFiles(directory=".", html=True), name="static")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        segments, _ = model.transcribe(tmp_path, beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
    finally:
        os.unlink(tmp_path)

    return {"text": text or "Aucun son détecté"}
