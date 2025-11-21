from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
import requests
import time
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cible le modèle Whisper TINY public de Hugging Face
# Nous utilisons un modèle sans clé pour tester la connexion (qualité faible, mais stable pour le test)
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-tiny"

@app.get("/")
async def serve_home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Erreur: index.html manquant</h1>", status_code=404)

@app.get("/style.css")
async def serve_css():
    try:
        with open("style.css", "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/css")
    except FileNotFoundError:
        return Response(status_code=404)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    audio_data = await file.read()
    headers = {} # Pas de clé API requise
    
    # Logique de Réessai pour gérer le modèle qui "dort" (503)
    for attempt in range(5):
        try:
            response = requests.post(API_URL, headers=headers, data=audio_data)
            
            # Cas 1 : Succès
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip() if isinstance(result, dict) else str(result)
                # On ajoute une mention claire que la connexion est bonne
                return {"text": f"✅ CONNEXION OK : {text}"} 

            # Cas 2 : Le modèle charge (Erreur 503)
            elif response.status_code == 503:
                estimated_time = response.json().get("estimated_time", 5)
                print(f"Le modèle dort. Attente de {estimated_time} secondes...")
                time.sleep(estimated_time)
                continue # On réessaie

            # Cas 3 : Autre erreur (404, 410, etc.)
            else:
                return {"text": f"❌ Erreur API ({response.status_code}) : Le serveur est injoignable."}

        except Exception as e:
            return {"text": f"❌ Erreur interne : {str(e)}"}

    return {"text": "❌ Échec de la connexion après plusieurs tentatives (Serveur saturé)."}
