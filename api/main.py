import os
import shutil
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from transformers import pipeline as hf_pipeline

# --- Imports internes ---
from src.processors.audio_engine import AudioEngine
from src.processors.nlp_engine import NLPEngine
from src.models.dl_model import InterviewModel
from src.models.ml_model import ScoringModel
from src.data_pipeline import DataPipeline
from src.evaluation.metrics import compute_soft_skills
from src.monitoring.metrics import FINAL_SCORE_GAUGE, API_REQUESTS
from database.db_manager import DBManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PRO8605 — Soft Skills Analysis API")

audio_engine = AudioEngine()
nlp_engine = NLPEngine()
dl_model = InterviewModel()
ml_model = ScoringModel()
db_manager = DBManager()

# --- LLM chargé une seule fois au démarrage ---
hf_token = os.getenv("HF_TOKEN")
try:
    llm_generator = hf_pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        token=hf_token if hf_token else None
    )
    logger.info("✅ TinyLlama chargé en mémoire.")
except Exception as _llm_err:
    llm_generator = None
    logger.warning(f"⚠️ TinyLlama non disponible au démarrage : {_llm_err}")

app.mount("/metrics", make_asgi_app())
app.mount("/static", StaticFiles(directory="api/static"), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse("api/static/index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


# ==========================================
# 🔧 ROUTES D'ENTRAÎNEMENT MLOPS
# ==========================================

@app.post("/train/dl", tags=["Training"])
async def train_dl_model():
    """Entraîne SimpleAudioNet (PyTorch) sur fake_sessions.csv."""
    try:
        pipeline_data = DataPipeline("storage/fake_sessions.csv")
        df = pipeline_data.extract()
        result = dl_model.train_custom_model(df, epochs=50, batch_size=16)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Erreur entraînement DL : {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/train/ml", tags=["Training"])
async def train_ml_model():
    """Entraîne le Random Forest Regressor sur fake_sessions.csv."""
    try:
        pipeline_data = DataPipeline("storage/fake_sessions.csv")
        df = pipeline_data.extract()
        result = ml_model.train(df)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Erreur entraînement ML : {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ==========================================
# 🎤 ROUTE D'INFÉRENCE PRINCIPALE
# ==========================================

@app.post("/analyze_file/", tags=["Analysis"])
async def analyze_file(file: UploadFile = File(...)):
    temp_path = None
    wav_path = None
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # Conversion en PCM WAV via ffmpeg (corrige PySoundFile/audioread avec WebM)
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 1. Extraction des caractéristiques audio
        audio_results = audio_engine.process_signal(wav_path)
        features = audio_results.get("features", {})

        # 2. Transcription (Whisper) + Émotion (SimpleAudioNet)
        transcription = dl_model.transcribe_audio(wav_path)
        emotion_data = dl_model.predict_emotion(audio_results["dl_input_vector"])

        # 3. Analyse NLP
        nlp_results = nlp_engine.analyze_text(transcription)

        # 4. Score global (Random Forest) + pénalités
        base_score = ml_model.predict_score(features, nlp_results, emotion_data)
        nb_fillers = nlp_results.get("filler_count", 0)
        penalty = nb_fillers * 10
        sentiment_penalty = 15 if nlp_results.get("sentiment_score", 0) < -0.1 else 0
        final_score = max(0, min(100, float(base_score) - penalty - sentiment_penalty))

        # Labels lisibles
        tempo_val = features.get("tempo", 0)
        if tempo_val < 90:
            label_tempo = "Lent"
        elif tempo_val < 130:
            label_tempo = "Modéré"
        else:
            label_tempo = "Rapide"

        vol_raw = features.get("volume", 0)
        vol_display = f"{round(vol_raw * 1000, 1)}%"

        sentiment_score = nlp_results.get("sentiment_score", 0)
        if sentiment_score > 0.1:
            sentiment_label = "Positif 😊"
        elif sentiment_score < -0.1:
            sentiment_label = "Négatif 😟"
        else:
            sentiment_label = "Neutre 😐"

        # 5. Résultat structuré
        soft_skills = compute_soft_skills(features, nlp_results)
        FINAL_SCORE_GAUGE.set(final_score)

        full_analysis = {
            "final_score": round(final_score, 2),
            "interpretation": (
                "Excellent" if final_score > 75
                else ("Moyen" if final_score > 45 else "À améliorer")
            ),
            "details": {
                "text_analysis": {
                    "transcription": transcription,
                    "sentiment": sentiment_label,
                    "fillers": nlp_results.get("fillers_details", {}),
                },
                "audio_analysis": {
                    "volume": vol_display,
                    "tempo_bpm": f"{round(tempo_val, 1)} ({label_tempo})",
                    "pause_ratio": f"{round(features.get('pause_ratio', 0) * 100, 1)}%",
                },
                "emotion_analysis": {
                    "label": emotion_data.get("emotion", "Neutre"),
                    "confidence": f"{round(emotion_data.get('confidence', 0) * 100, 1)}%",
                },
                "soft_skills": {
                    "stress": round(soft_skills["stress"], 3),
                    "confidence": round(soft_skills["confidence"], 3),
                    "dynamism": round(soft_skills["dynamism"], 3),
                },
            },
        }

        # 6. Rapport LLM (TinyLlama — chargé au démarrage)
        try:
            if llm_generator is None:
                raise RuntimeError("LLM non chargé au démarrage")

            score = full_analysis["final_score"]
            fillers = full_analysis["details"]["text_analysis"]["fillers"]
            filler_count = sum(fillers.values()) if isinstance(fillers, dict) else 0

            prompt_instruction = (
                "<|system|>Tu es un assistant RH expert en analyse d'entretiens.<|end|>\n"
                "<|user|>Génère un rapport détaillé comprenant :\n"
                "1. Résumé global\n"
                "2. Analyse des scores\n"
                "3. Analyse de la transcription\n"
                "4. Feedback personnalisé (points forts / axes d'amélioration)\n"
                "5. Conseils pratiques\n\n"
                f"Score final : {score}\n"
                f"Sentiment : {sentiment_label}\n"
                f"Émotion : {emotion_data.get('emotion', 'Neutre')}\n"
                f"Volume : {vol_display}\n"
                f"Tempo : {round(tempo_val, 1)} BPM ({label_tempo})\n"
                f"Ratio de pauses : {round(features.get('pause_ratio', 0) * 100, 1)}%\n"
                f"Tics de langage détectés : {filler_count}\n"
                f"Transcription : {transcription}\n"
                "Sois factuel, bienveillant et pédagogique.<|end|>\n<|assistant|>"
            )
            outputs = llm_generator(
                prompt_instruction,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=2,
            )
            rapport = outputs[0]["generated_text"].split("<|assistant|>")[-1].strip()
            full_analysis["llm_report"] = rapport
        except Exception as e:
            logger.error(f"Erreur génération rapport LLM : {e}")
            full_analysis["llm_report"] = "Rapport LLM non disponible."

        # 7. Persistance en base de données
        try:
            db_manager.save_session({
                "duration": audio_results.get("meta", {}).get("duration"),
                "sentiment_score": nlp_results.get("sentiment_score"),
                "pause_ratio": features.get("pause_ratio"),
                "transcription": transcription,
                "final_score": final_score,
                "emotion": emotion_data.get("emotion"),
                "filler_count": nlp_results.get("filler_count"),
            })
        except Exception as e:
            logger.warning(f"Sauvegarde DB échouée (non bloquant) : {e}")

        API_REQUESTS.labels(endpoint="/analyze_file/", status="success").inc()
        return JSONResponse(content=full_analysis)

    except Exception as e:
        logger.error(f"Erreur analyse : {e}")
        API_REQUESTS.labels(endpoint="/analyze_file/", status="error").inc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)