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

# --- LISTE DE SECOURS ---
# Le code va essayer ces modèles dans l'ordre jusqu'à ce qu'un marche.
MODELS = [
    "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo",
    "https://api-inference.huggingface.co/models/openai/whisper-large-v3",
    "https://api-inference.huggingface.co/models/openai/whisper-medium",
    "https://api-inference.huggingface.co/models/openai/whisper-base",
    "https://api-inference.huggingface.co/models/openai/whisper-tiny"
]

HF_API_KEY = os.environ.get("HF_API_KEY")

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
    if not HF_API_KEY:
        return {"text": "❌ Erreur : Clé API manquante."}

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    audio_data = await file.read()

    # On teste chaque modèle de la liste
    for model_url in MODELS:
        print(f"Tentative avec : {model_url.split('/')[-1]}...")
        
        # On essaie 3 fois par modèle (au cas où il dort - erreur 503)
        for attempt in range(3):
            try:
                response = requests.post(model_url, headers=headers, data=audio_data)
                
                # SUCCÈS
                if response.status_code == 200:
                    result = response.json()
                    text = result.get("text", "") if isinstance(result, dict) else result[0].get("text", "")
                    return {"text": text.strip()}
                
                # MODÈLE DORT (503) -> On attend et on réessaie le MÊME modèle
                elif response.status_code == 503:
                    wait_time = response.json().get("estimated_time", 5)
                    print(f"   -> Dort... attente {wait_time}s")
                    time.sleep(wait_time)
                    continue 

                # ERREUR FATALE (410 Gone, 404, etc.) -> On casse la boucle pour essayer le modèle SUIVANT
                else:
                    print(f"   -> Erreur {response.status_code}. Passage au modèle suivant.")
                    break 

            except Exception as e:
                print(f"   -> Erreur connection: {e}")
                break
        
        # Si on est ici, c'est que ce modèle a échoué, la boucle 'for model_url' continue vers le suivant

    return {"text": "❌ Tous les modèles sont indisponibles pour le moment. Réessayez dans 1h."}
