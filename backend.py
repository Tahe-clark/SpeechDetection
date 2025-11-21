from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
import requests
import os
import time
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration Replicate
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY")

# Le modèle Whisper le plus récent sur Replicate
REPLICATE_MODEL_URL = "https://api.replicate.com/v1/predictions"
MODEL_ID = "openai/whisper:4676be329c299c824c8b355d142d2427b329ef31a9667f3743c39175a22c5496"

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
    if not REPLICATE_API_KEY:
        return {"text": "❌ Erreur : Clé Replicate manquante."}

    audio_content = await file.read()
    
    # Encoder l'audio en Base64 (nécessaire pour Replicate)
    base64_audio = base64.b64encode(audio_content).decode('utf-8')
    data_uri = f"data:{file.content_type};base64,{base64_audio}"

    headers = {
        "Authorization": f"Token {REPLICATE_API_KEY}",
        "Content-Type": "application/json"
    }

    # Créer la prédiction
    response = requests.post(
        REPLICATE_MODEL_URL,
        headers=headers,
        json={
            "version": MODEL_ID,
            "input": {
                "audio": data_uri,
                "transcription": "french" # Précision de la langue
            }
        }
    )

    if response.status_code != 201:
        return {"text": f"❌ Erreur Replicate (Start): {response.status_code} - {response.text}"}

    prediction_id = response.json().get('id')
    
    # Poll (attendre) le résultat
    status = "starting"
    while status not in ["succeeded", "failed"]:
        await time.sleep(2) # Attendre 2 secondes entre les requêtes
        
        response = requests.get(
            f"{REPLICATE_MODEL_URL}/{prediction_id}",
            headers=headers
        )
        
        if response.status_code != 200:
            return {"text": f"❌ Erreur Replicate (Poll): {response.status_code}"}
        
        data = response.json()
        status = data.get('status')
        print(f"Statut Replicate: {status}")

    if status == "succeeded":
        # Le texte est dans un format spécifique pour ce modèle
        text = data['output']['text'].strip() if data.get('output') and 'text' in data['output'] else "Aucun texte transcrit."
        return {"text": text}
    else:
        error = data.get('error', 'Erreur inconnue.')
        return {"text": f"❌ Échec de la transcription (Replicate): {error}"}
