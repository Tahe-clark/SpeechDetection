from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
import requests
import os
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Configuration Hugging Face (Whisper Small)
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
HF_API_KEY = os.environ.get("HF_API_KEY")

# 2. ROUTE D'ACCUEIL : C'est ce qui manquait ! 
# Quand on ouvre le site, on lit et on affiche index.html
@app.get("/")
async def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Erreur: index.html introuvable</h1>", status_code=404)

# 3. ROUTE STYLE : Pour que le site soit joli
@app.get("/style.css")
async def serve_css():
    try:
        with open("style.css", "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/css")
    except FileNotFoundError:
        return Response(status_code=404)

# 4. ROUTE TRANSCRIPTION : La logique de traduction
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not HF_API_KEY:
        return {"text": "❌ Erreur : Clé API manquante dans Render."}

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    audio_data = await file.read()

    # Système de réessai automatique (5 tentatives)
    for attempt in range(5):
        try:
            response = requests.post(API_URL, headers=headers, data=audio_data)
            
            if response.status_code == 200:
                result = response.json()
                return {"text": result.get("text", "").strip() if isinstance(result, dict) else str(result)}
            
            elif response.status_code == 503:
                wait_time = response.json().get("estimated_time", 10)
                print(f"Le modèle charge... attente de {wait_time}s")
                time.sleep(wait_time)
                continue # On réessaie
            
            else:
                return {"text": f"❌ Erreur API: {response.status_code}"}
        except Exception as e:
            return {"text": f"❌ Erreur interne: {str(e)}"}
            
    return {"text": "❌ Trop de tentatives, réessayez plus tard."}
