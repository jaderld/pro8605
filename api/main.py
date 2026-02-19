import os
import shutil
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

# --- Imports de ton intelligence ---
from src.processors.audio_engine import AudioEngine
from src.processors.nlp_engine import NLPEngine
from src.models.dl_model import InterviewModel
from src.models.ml_model import ScoringModel
from src.monitoring.metrics import PROCESSING_TIME
from src.monitoring.metrics import FINAL_SCORE_GAUGE

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="InterviewFlow AI API", 
    description="API MLOps avec Interface Web intégrée", 
    version="1.0"
)

# --- INITIALISATION DES MOTEURS ---
audio_engine = AudioEngine()
nlp_engine = NLPEngine()
dl_model = InterviewModel() 
ml_model = ScoringModel()

# --- MÉTRIQUES PROMETHEUS ---
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# --- INTERFACE WEB STATIQUE (Le Frontend intégré) ---
app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/", tags=["UI"])
async def serve_frontend():
    """Sert la page HTML principale de l'application"""
    return FileResponse('api/static/index.html')

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "models_loaded": True}

# --- PIPELINE D'ANALYSE PRINCIPAL ---
@app.post("/analyze_file/", tags=["Analysis"])
async def analyze_file(file: UploadFile = File(...)):
    """Reçoit l'audio du navigateur, l'analyse, et renvoie le JSON de résultats."""
    temp_path = None
    try:
        # 1. Sauvegarde temporaire du fichier audio entrant
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        with PROCESSING_TIME.labels(module='total').time():
            # 2. Audio Engine : Analyse physique (Pause ratio, Volume, BPM)
            # On utilise l'audio complet pour calculer les stats de silence
            audio_results = audio_engine.process_signal(temp_path)
            
            if audio_results.get('status') == 'error':
                raise Exception(audio_results.get('error', 'Erreur audio interne'))

            # 3. Deep Learning : Transcription Whisper + Émotion PyTorch
            # NOTE : On envoie le temp_path DIRECTEMENT sans segments pour que Whisper
            # entende les "euh" et les hésitations.
            transcription = dl_model.transcribe_audio(temp_path)
            
            # Analyse d'émotion basée sur le vecteur de caractéristiques audio
            emotion_data = dl_model.predict_emotion(audio_results['dl_input_vector'])

            # 4. NLP Engine : Analyse du texte (Comptage des tics et sentiment)
            nlp_results = nlp_engine.analyze_text(transcription)

            # 5. ML Model : Calcul de la note globale (Juge final)
            # On passe le dictionnaire 'features' (contient pause_ratio, volume, etc.)
            final_score = ml_model.predict_score(audio_results['features'], nlp_results)
            FINAL_SCORE_GAUGE.set(final_score)

            # 6. Assemblage du résultat final
            full_analysis = {
                "filename": file.filename,
                "transcription": transcription,
                "acoustics": audio_results['features'],
                "nlp": nlp_results,
                "emotion_analysis": emotion_data,
                "final_scoring": {
                    "overall_score": round(float(final_score), 2),
                    "interpretation": "Excellent" if final_score > 80 else "À améliorer"
                }
            }

            logger.info(f"Analyse terminée pour {file.filename} - Score: {final_score}")
            return JSONResponse(content=full_analysis)

    except Exception as e:
        logger.error(f"Erreur d'analyse : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Nettoyage du fichier temporaire pour éviter de saturer Docker
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# --- ROUTES MLOPS (Entraînement) ---
@app.post("/ml/train/", tags=["MLOps"])
def train_scoring_model():
    import pandas as pd
    try:
        df = pd.read_csv('storage/fake_sessions.csv')
        ml_model.train(df)
        return {"status": "trained", "message": "Modèle RandomForest mis à jour avec succès."}
    except Exception as e:
        return {"status": "Error", "error": str(e)}

@app.post("/dl/train/", tags=["MLOps"])
def train_emotion_model():
    # Placeholder pour l'entraînement PyTorch futur
    return {"status": "Success", "message": "Réseau PyTorch prêt pour entraînement."}