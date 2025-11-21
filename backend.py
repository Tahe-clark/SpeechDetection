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

# --- MODIFICATION ICI ---
# On utilise le modèle "Turbo" car le "Small" renvoie une erreur 410 (n'est plus dispo).
# Le Turbo est gratuit, très rapide et fonctionne actuellement.
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
HF_API_KEY = os.environ.get("HF_API_KEY")

# Route Accueil
@app.get("/")
async def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Erreur: index.html introuvable</h1>", status_code=404)

# Route CSS
@app.get("/style.css")
async def serve_css():
    try:
        with open("style.css", "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/css")
    except FileNotFoundError:
        return Response(status_code=404)

# Route Transcription
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not HF_API_KEY:
        return {"text": "❌ Erreur : Clé API manquante dans Render."}

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    audio_data = await file.read()

    # Boucle de sécurité (5 essais)
    for attempt in range(5):
        try:
            response = requests.post(API_URL, headers=headers, data=audio_data)
            
            if response.status_code == 200:
                result = response.json()
                # Parfois le résultat est direct, parfois dans une liste
                text = result.get("text", "") if isinstance(result, dict) else result[0].get("text", "")
                return {"text": text.strip()}
            
            elif response.status_code == 503:
                # Le modèle dort, on attend
                wait_time = response.json().get("estimated_time", 10)
                print(f"Modèle en chargement... {wait_time}s")
                time.sleep(wait_time)
                continue
            
            else:
                # Si on a une 410 ou autre, on l'affiche
                return {"text": f"❌ Erreur API: {response.status_code} (Modèle indisponible)"}

        except Exception as e:
            return {"text": f"❌ Erreur interne: {str(e)}"}
            
    return {"text": "❌ Délai dépassé, réessayez."}
