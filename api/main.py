import os
import shutil
import tempfile
import logging
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from transformers import pipeline

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
# ROUTES D'ENTRAÎNEMENT MLOPS
# ==========================================

# ==========================================
# ROUTE LLM
# ==========================================
from fastapi import Body


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


        # 5. RÉSULTAT FINAL (sans rapport LLM)
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

        # 6. Génération automatique du rapport LLM

        try:
            from transformers import pipeline
            score = full_analysis.get('final_score', '--')
            sentiment = full_analysis.get('details', {}).get('text_analysis', {}).get('sentiment', '--')
            stress = full_analysis.get('details', {}).get('emotion_analysis', {}).get('label', '--')
            volume = full_analysis.get('details', {}).get('audio_analysis', {}).get('volume', '--')
            tempo = full_analysis.get('details', {}).get('audio_analysis', {}).get('tempo_bpm', '--')
            pause_ratio = full_analysis.get('details', {}).get('audio_analysis', {}).get('pause_ratio', '--')
            fillers = full_analysis.get('details', {}).get('text_analysis', {}).get('fillers', {})
            filler_count = sum(fillers.values()) if isinstance(fillers, dict) else fillers or 0
            transcription = full_analysis.get('details', {}).get('text_analysis', {}).get('transcription', '--')

            prompt = f"""
Tu es un assistant RH expert en analyse d’entretiens.
À partir des données suivantes issues d’un entretien simulé (scores, métriques, transcription), génère un rapport détaillé et professionnel comprenant :

1. Résumé global
2. Analyse des scores (avec explications)
3. Analyse de la transcription
4. Feedback personnalisé (points forts, axes d’amélioration, points problématiques)
5. Conseils pratiques

Données d’entrée :
- Score final : {score}
- Sentiment : {sentiment}
- Stress détecté : {stress}
- Volume : {volume}
- Tempo : {tempo}
- Ratio de pauses : {pause_ratio}
- Nombre de tics de langage : {filler_count}
- Transcription : {transcription}

Sois factuel, bienveillant, pédagogique et justifie chaque remarque.
"""
            # Utilisation d'un modèle instruction-following (TinyLlama ou Zephyr)
            # Pour TinyLlama : "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            # Pour Zephyr : "HuggingFaceH4/zephyr-7b-beta"
            # On prend TinyLlama pour la légèreté (modèle 1.1B, fonctionne sur CPU)
            hf_token = os.getenv("HF_TOKEN")
            generator = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                # tokenizer n'est pas obligatoire si auto-détecté, sinon :
                # tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                use_auth_token=hf_token if hf_token else None
            )
            # Format prompt type chat/instruction
            prompt_instruction = f"""<|system|>Tu es un assistant RH expert en analyse d’entretiens.<|end|>\n<|user|>À partir des données suivantes issues d’un entretien simulé (scores, métriques, transcription), génère un rapport détaillé et professionnel comprenant :\n1. Résumé global\n2. Analyse des scores (avec explications)\n3. Analyse de la transcription\n4. Feedback personnalisé (points forts, axes d’amélioration, points problématiques)\n5. Conseils pratiques\n\nDonnées d’entrée :\n- Score final : {score}\n- Sentiment : {sentiment}\n- Stress détecté : {stress}\n- Volume : {volume}\n- Tempo : {tempo}\n- Ratio de pauses : {pause_ratio}\n- Nombre de tics de langage : {filler_count}\n- Transcription : {transcription}\n\nSois factuel, bienveillant, pédagogique et justifie chaque remarque.<|end|>\n<|assistant|>"""
            outputs = generator(prompt_instruction)
            rapport = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
            full_analysis["llm_report"] = rapport
        except Exception as e:
            logger.error(f"Erreur génération rapport LLM : {str(e)}")
            full_analysis["llm_report"] = "Erreur lors de la génération du rapport LLM."

        return JSONResponse(content=full_analysis)

    except Exception as e:
        logger.error(f"Erreur : {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)