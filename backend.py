from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
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
model = whisper.load_model("small.en")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    suffix = ".webm" if "webm" in file.content_type else ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    try:
        result = model.transcribe(tmp.name, language="en", fp16=False)
        text = result["text"].strip()
    except Exception as e:
        text = f"Erreur : {e}"
    finally:
        os.unlink(tmp.name)

    return {"text": text}

