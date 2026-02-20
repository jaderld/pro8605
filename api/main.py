import os
import shutil
import tempfile
import logging
import pandas as pd # <-- AJOUTÉ POUR LES ROUTES TRAIN
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

# --- Imports internes ---
from src.processors.audio_engine import AudioEngine
from src.processors.nlp_engine import NLPEngine
from src.models.dl_model import InterviewModel
from src.models.ml_model import ScoringModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="InterviewFlow AI API")

audio_engine = AudioEngine()
nlp_engine = NLPEngine()
dl_model = InterviewModel() 
ml_model = ScoringModel()

app.mount("/metrics", make_asgi_app())
app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse('api/static/index.html')

# ==========================================
# 🚀 ROUTES D'ENTRAÎNEMENT MLOPS (AJOUTÉES)
# ==========================================

@app.post("/ml/train/", tags=["MLOps"])
async def train_ml_model():
    """Déclenche le ré-entraînement du Random Forest (Score final)"""
    try:
        csv_path = 'storage/fake_sessions.csv'
        if not os.path.exists(csv_path):
            return JSONResponse(status_code=404, content={"error": "Fichier dataset introuvable."})
            
        df = pd.read_csv(csv_path)
        metrics = ml_model.train(df)
        
        return {"message": "Random Forest ré-entraîné avec succès !", "metrics": metrics}
    except Exception as e:
        logger.error(f"Erreur Entraînement ML : {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/dl/train/", tags=["MLOps"])
async def train_dl_model():
    """Déclenche le ré-entraînement du modèle PyTorch (Émotion binaire)"""
    try:
        csv_path = 'storage/fake_sessions.csv'
        if not os.path.exists(csv_path):
            return JSONResponse(status_code=404, content={"error": "Fichier dataset introuvable."})
            
        df = pd.read_csv(csv_path)
        # On appelle la fonction d'entraînement de PyTorch
        metrics = dl_model.train_custom_model(df)
        
        return {"message": "Modèle PyTorch (Binaire) ré-entraîné avec succès !", "metrics": metrics}
    except Exception as e:
        logger.error(f"Erreur Entraînement DL : {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ==========================================
# 🎤 ROUTE D'INFÉRENCE PRINCIPALE
# ==========================================

@app.post("/analyze_file/", tags=["Analysis"])
async def analyze_file(file: UploadFile = File(...)):
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # 1. Extraction des caractéristiques
        audio_results = audio_engine.process_signal(temp_path)
        features = audio_results.get('features', {})
        
        # 2. Transcription et Émotion
        transcription = dl_model.transcribe_audio(temp_path)
        emotion_data = dl_model.predict_emotion(audio_results['dl_input_vector'])

        # 3. NLP
        nlp_results = nlp_engine.analyze_text(transcription)

        # 4. CALCUL DU SCORE AVEC PÉNALITÉS
        base_score = ml_model.predict_score(features, nlp_results, emotion_data)
        nb_fillers = nlp_results.get('filler_count', 0)
        penalty = nb_fillers * 10
        sentiment_penalty = 15 if nlp_results.get('sentiment_score', 0) < -0.1 else 0
        final_score = max(0, min(100, float(base_score) - penalty - sentiment_penalty))

        # --- GESTION DU BPM ---
        tempo_val = features.get('tempo', 0)
        if tempo_val < 90: label_tempo = "Lent"
        elif tempo_val < 130: label_tempo = "Modéré"
        else: label_tempo = "Rapide"

        # --- GESTION DU VOLUME ---
        vol_raw = features.get('volume', 0)
        vol_percent = round(vol_raw * 1000, 1)
        vol_display = f"{vol_percent}%"

        # 5. RÉSULTAT FINAL
        full_analysis = {
            "final_score": round(final_score, 2),
            "interpretation": "Excellent" if final_score > 75 else ("Moyen" if final_score > 45 else "À améliorer"),
            "details": {
                "text_analysis": {
                    "transcription": transcription,
                    "sentiment": "Positif 😊" if nlp_results.get('sentiment_score', 0) > 0.1 else ("Négatif 😟" if nlp_results.get('sentiment_score', 0) < -0.1 else "Neutre 😐"),
                    "fillers": nlp_results.get('fillers_details', {})
                },
                "audio_analysis": {
                    "volume": vol_display,  
                    "tempo_bpm": f"{round(tempo_val, 1)} ({label_tempo})", 
                    "pause_ratio": f"{round(features.get('pause_ratio', 0) * 100, 1)}%"
                },
                "emotion_analysis": {
                    "label": emotion_data.get('emotion', 'Neutre'),
                }
            }
        }
        return JSONResponse(content=full_analysis)

    except Exception as e:
        logger.error(f"Erreur : {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)