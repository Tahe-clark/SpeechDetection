from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os

from fastapi.staticfiles import StaticFiles  # ← AJOUTE

app = FastAPI()
app.mount("/", StaticFiles(directory=".", html=True), name="static")  # ← AJOUTE

# Autorise ton navigateur local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle rapide (avec une précision de +30% en anglais)
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    suffix = ".webm" if "webm" in file.content_type else ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    try:
        segments, _ = model.transcribe(tmp.name, language="en", beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        text = f"Erreur Whisper : {str(e)}"
    finally:
        os.unlink(tmp.name)

    return {"text": text or "Aucun son détecté"}

