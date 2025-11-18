from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from io import BytesIO

app = FastAPI()

# Sert le frontend
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# Autorise tous les origines (téléphone, navigateur, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle ultra-léger et rapide (fonctionne sur Render/Railway gratuit)
model = WhisperModel("small.en", device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    content = await file.read()
    audio = BytesIO(content)
    
    segments, _ = model.transcribe(audio, language="en", beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()
    
    return {"text": text or "Aucun son détecté"}
