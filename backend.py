from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import time

app = FastAPI()

# Autoriser le frontend à parler au backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# URL Spécifique pour le modèle SMALL
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"
HF_API_KEY = os.environ.get("HF_API_KEY")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not HF_API_KEY:
        return {"text": "❌ Erreur serveur : Clé API manquante."}

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    audio_data = await file.read()

    # --- Logique de Réessai (Anti-Plantages) ---
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=audio_data)
            
            # Cas 1 : Tout va bien
            if response.status_code == 200:
                result = response.json()
                # Parfois le texte est dans une liste, parfois dans un dict
                if isinstance(result, dict):
                    return {"text": result.get("text", "").strip()}
                return {"text": str(result)}

            # Cas 2 : Le modèle charge (Erreur 503)
            elif response.status_code == 503:
                estimated_time = response.json().get("estimated_time", 10)
                print(f"Le modèle dort. Attente de {estimated_time} secondes...")
                time.sleep(estimated_time) # On attend la durée demandée par HF
                continue # On recommence la boucle

            # Cas 3 : Autre erreur
            else:
                return {"text": f"❌ Erreur API ({response.status_code})"}

        except Exception as e:
            return {"text": f"❌ Erreur interne : {str(e)}"}

    return {"text": "❌ Le serveur est trop occupé, réessayez plus tard."}
