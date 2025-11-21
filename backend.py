from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
import google.generativeai as genai
import os
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration Google Gemini
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

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
    if not GOOGLE_API_KEY:
        return {"text": "❌ Erreur : Clé Google manquante."}

    # Gemini a besoin d'un fichier physique temporaire pour bien traiter l'audio
    suffix = ".webm" if "webm" in file.content_type else ".wav"
    
    try:
        # 1. On crée un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 2. On charge le fichier pour Google
        audio_file = genai.upload_file(tmp_path)

        # 3. On utilise le modèle PRO (Le plus intelligent pour les voix difficiles)
        model = genai.GenerativeModel("gemini-1.5-pro")

        # 4. Le PROMPT MAGIQUE : On explique à l'IA quoi faire
        # C'est ici que Gemini bat Whisper : on lui donne du contexte.
        prompt = """
        Tu es un assistant expert en accessibilité et orthophonie.
        Ta mission est de transcrire cet enregistrement audio en texte français.
        IMPORTANT : La personne qui parle a des difficultés d'articulation (voix altérée).
        - Utilise le contexte pour deviner les mots mal prononcés.
        - Rétablis une syntaxe correcte si nécessaire pour que la phrase ait du sens.
        - Ne réponds QUE par le texte transcrit, sans ajouter de commentaires ni de guillemets.
        """

        response = model.generate_content([prompt, audio_file])
        
        # Nettoyage du fichier temporaire
        os.unlink(tmp_path)
        
        return {"text": response.text.strip()}

    except Exception as e:
        return {"text": f"❌ Erreur Google : {str(e)}"}
