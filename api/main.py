import os
import json
import shutil
import tempfile
import logging
import yaml
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

# --- Imports internes ---
from src.processors.audio_engine import AudioEngine
from src.processors.nlp_engine import NLPEngine
from src.models.dl_model import InterviewModel
from src.models.ml_model import ScoringModel
from src.data_pipeline import DataPipeline
from src.monitoring.metrics import FINAL_SCORE_GAUGE, BASE_SCORE_GAUGE, SCORE_PENALTY, SCORE_INTERPRETATION, API_REQUESTS, PROCESSING_TIME
from database.db_manager import DBManager
from src.processors.report_generator import generate_structured_report
from src.processors.ollama_client import generate_conclusion, generate_conclusion_stream, generate_interview_question, generate_relevance_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PRO8605 — Soft Skills Analysis API")

audio_engine = AudioEngine()
nlp_engine = NLPEngine()
dl_model = InterviewModel()
ml_model = ScoringModel()

# --- Base de données : PostgreSQL (settings.yaml) avec fallback SQLite ---
try:
    with open("config/settings.yaml", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)
    _pg = _cfg.get("postgres", {})
    db_manager = DBManager(
        use_postgres=True,
        pg_config={
            "host": os.getenv("POSTGRES_HOST", _pg.get("host", "postgres")),
            "port": int(os.getenv("POSTGRES_PORT", _pg.get("port", 5432))),
            "user": os.getenv("POSTGRES_USER", _pg.get("user", "interviewuser")),
            "password": os.getenv("POSTGRES_PASSWORD", _pg.get("password", "interviewpass")),
            "dbname": os.getenv("POSTGRES_DB", _pg.get("dbname", "pro8605")),
        },
    )
    logger.info("✅ DBManager connecté à PostgreSQL.")
except Exception as _db_err:
    logger.warning(f"⚠️ PostgreSQL indisponible, fallback SQLite : {_db_err}")
    db_manager = DBManager()

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
# 💬 GÉNÉRATION DE QUESTION D'ENTRETIEN (LLM)
# ==========================================

@app.post("/generate_question/", tags=["Interview"])
async def generate_question(
    domain: str = Form(...),
    position: str = Form(...),
    focus_points: str = Form(""),
):
    """Génère une question d'entretien personnalisée via Ollama."""
    try:
        question = generate_interview_question(
            domain=domain,
            position=position,
            focus_points=focus_points,
        )
        if question:
            return JSONResponse(content={"question": question})
        return JSONResponse(
            status_code=503,
            content={"error": "Le LLM n'est pas disponible. Veuillez réessayer."},
        )
    except Exception as e:
        logger.error(f"Erreur génération question : {e}")
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

        # Calcul du vrai débit de parole (WPM) depuis la transcription + durée de parole
        # Remplace le BPM librosa (rythme musical, inutile sur la parole)
        speech_segments = audio_results.get("speech_segments", [])
        speech_duration = sum(s["end_s"] - s["start_s"] for s in speech_segments)
        effective_duration = max(1.0, speech_duration if speech_duration > 0 else audio_results.get("meta", {}).get("duration", 1.0))
        wpm = round((max(1, nlp_results.get("word_count", 1)) / effective_duration) * 60, 1)
        features["tempo"] = wpm  # override BPM librosa par WPM réel

        # 4. Score global (Random Forest)
        # Le RF a appris une formule pondérée (30% tics normalisés, 20% débit WPM, 15% sentiment…)
        # directement depuis les données simulées → aucune pénalité post-hoc nécessaire.
        rf_score = max(0, min(100, round(float(ml_model.predict_score(features, nlp_results, emotion_data)), 2)))

        # ── Gardes-fous hors distribution ─────────────────────────────────
        # Le RF ne sait pas extrapoler sur des valeurs extrêmes jamais vues à l'entraînement.
        # Ces plafonds durs s'appliquent uniquement aux cas objectivement rédhibitoires.
        word_count_check = max(1, nlp_results.get("word_count", 1))
        filler_rate_check = nlp_results.get("filler_count", 0) / word_count_check

        if filler_rate_check >= 0.30:
            final_score = min(rf_score, 25)    # > 30% de tics → 1 mot sur 3 est un tic
        elif filler_rate_check >= 0.20:
            final_score = min(rf_score, 45)    # > 20% de tics → très préoccupant
        elif filler_rate_check >= 0.12:
            final_score = min(rf_score, 65)    # > 12% de tics → au-dessus du seuil critique
        elif wpm > 220:
            final_score = min(rf_score, 55)    # > 220 mots/min → débit incompréhensible
        elif wpm > 190:
            final_score = min(rf_score, 72)    # > 190 mots/min → trop rapide
        else:
            final_score = rf_score

        # Métriques Prometheus étape scoring
        BASE_SCORE_GAUGE.set(rf_score)
        SCORE_PENALTY.observe(max(0, rf_score - final_score))
        PROCESSING_TIME.labels(module='scoring').observe(0)  # instantané

        # Labels lisibles — seuils WPM (mots par minute)
        tempo_val = wpm
        if tempo_val < 110:
            label_tempo = "Lent"
        elif tempo_val <= 170:
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
        if final_score > 75:
            interpretation = "Excellent"
        elif final_score > 45:
            interpretation = "Moyen"
        elif final_score > 20:
            interpretation = "À améliorer"
        else:
            interpretation = "Insuffisant"
        FINAL_SCORE_GAUGE.set(final_score)
        SCORE_INTERPRETATION.labels(interpretation=interpretation).inc()

        full_analysis = {
            "final_score": round(final_score, 2),
            "interpretation": interpretation,
            "details": {
                "text_analysis": {
                    "transcription": transcription,
                    "sentiment": sentiment_label,
                    "fillers": nlp_results.get("fillers_details", {}),
                },
                "audio_analysis": {
                    "volume": vol_display,
                    "tempo_bpm": f"{round(tempo_val, 1)} mots/min ({label_tempo})",
                    "pause_ratio": f"{round(features.get('pause_ratio', 0) * 100, 1)}%",
                },
                "emotion_analysis": {
                    "label": emotion_data.get("emotion", "Neutre"),
                    "confidence": f"{round(emotion_data.get('confidence', 0) * 100, 1)}%",
                },
            },
        }

        # 6. Rapport structuré (générateur basé sur les règles métier) + conclusion LLM (Ollama)
        try:
            fillers = full_analysis["details"]["text_analysis"]["fillers"]
            filler_count_report = sum(fillers.values()) if isinstance(fillers, dict) else 0

            # Étape A — génération des 5 sections déterministes
            # (retourne aussi points_forts et axes pour construire le prompt Ollama)
            rapport_base, points_forts, axes = generate_structured_report(
                score=full_analysis["final_score"],
                interpretation=full_analysis["interpretation"],
                sentiment_label=sentiment_label,
                emotion=emotion_data.get("emotion", "Neutre"),
                vol_raw=vol_raw,
                vol_display=vol_display,
                tempo_val=tempo_val,
                label_tempo=label_tempo,
                pause_ratio_pct=round(features.get("pause_ratio", 0) * 100, 1),
                filler_count=filler_count_report,
                fillers_dict=fillers if isinstance(fillers, dict) else {},
                transcription=transcription,
                word_count=nlp_results.get("word_count", 0),
                llm_conclusion=None,  # pas encore disponible
            )

            # Étape B — conclusion narrative via Ollama (non bloquant si indisponible)
            llm_conclusion = generate_conclusion(
                score=full_analysis["final_score"],
                interpretation=full_analysis["interpretation"],
                emotion=emotion_data.get("emotion", "Neutre"),
                sentiment_label=sentiment_label,
                filler_count=filler_count_report,
                tempo_val=tempo_val,
                pause_ratio_pct=round(features.get("pause_ratio", 0) * 100, 1),
                word_count=nlp_results.get("word_count", 0),
                points_forts=points_forts,
                axes=axes,
            )

            # Étape C — si Ollama a répondu, on régénère avec la section 6
            if llm_conclusion:
                rapport, _, _ = generate_structured_report(
                    score=full_analysis["final_score"],
                    interpretation=full_analysis["interpretation"],
                    sentiment_label=sentiment_label,
                    emotion=emotion_data.get("emotion", "Neutre"),
                    vol_raw=vol_raw,
                    vol_display=vol_display,
                    tempo_val=tempo_val,
                    label_tempo=label_tempo,
                    pause_ratio_pct=round(features.get("pause_ratio", 0) * 100, 1),
                    filler_count=filler_count_report,
                    fillers_dict=fillers if isinstance(fillers, dict) else {},
                    transcription=transcription,
                    word_count=nlp_results.get("word_count", 0),
                    llm_conclusion=llm_conclusion,
                )
            else:
                rapport = rapport_base  # Ollama indisponible : rapport 5 sections suffit

            full_analysis["llm_report"] = rapport
        except Exception as e:
            logger.error(f"Erreur génération rapport : {e}")
            full_analysis["llm_report"] = "Rapport non disponible."

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


# ==========================================
# 🎤 ROUTE SSE — SCORES IMMÉDIATS + RAPPORT EN STREAMING
# ==========================================

@app.post("/analyze_stream/", tags=["Analysis"])
async def analyze_stream(
    file: UploadFile = File(...),
    interview_question: str = Form(""),
    interview_domain: str = Form(""),
    interview_position: str = Form(""),
):
    """
    Endpoint SSE : renvoie d'abord les scores (event: scores),
    puis le rapport ligne par ligne (event: report_chunk),
    puis la conclusion LLM token par token (event: llm_token),
    puis l'analyse de pertinence question/réponse (event: relevance_token),
    et enfin un signal de fin (event: done).
    """
    temp_path = None
    wav_path = None

    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # 1-5 : pipeline identique à /analyze_file/
        audio_results = audio_engine.process_signal(wav_path)
        features = audio_results.get("features", {})

        transcription = dl_model.transcribe_audio(wav_path)
        emotion_data = dl_model.predict_emotion(audio_results["dl_input_vector"])
        nlp_results = nlp_engine.analyze_text(transcription)

        speech_segments = audio_results.get("speech_segments", [])
        speech_duration = sum(s["end_s"] - s["start_s"] for s in speech_segments)
        effective_duration = max(1.0, speech_duration if speech_duration > 0 else audio_results.get("meta", {}).get("duration", 1.0))
        wpm = round((max(1, nlp_results.get("word_count", 1)) / effective_duration) * 60, 1)
        features["tempo"] = wpm

        rf_score = max(0, min(100, round(float(ml_model.predict_score(features, nlp_results, emotion_data)), 2)))

        word_count_check = max(1, nlp_results.get("word_count", 1))
        filler_rate_check = nlp_results.get("filler_count", 0) / word_count_check

        if filler_rate_check >= 0.30:
            final_score = min(rf_score, 25)
        elif filler_rate_check >= 0.20:
            final_score = min(rf_score, 45)
        elif filler_rate_check >= 0.12:
            final_score = min(rf_score, 65)
        elif wpm > 220:
            final_score = min(rf_score, 55)
        elif wpm > 190:
            final_score = min(rf_score, 72)
        else:
            final_score = rf_score

        BASE_SCORE_GAUGE.set(rf_score)
        SCORE_PENALTY.observe(max(0, rf_score - final_score))
        PROCESSING_TIME.labels(module='scoring').observe(0)

        tempo_val = wpm
        if tempo_val < 110:
            label_tempo = "Lent"
        elif tempo_val <= 170:
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

        if final_score > 75:
            interpretation = "Excellent"
        elif final_score > 45:
            interpretation = "Moyen"
        elif final_score > 20:
            interpretation = "À améliorer"
        else:
            interpretation = "Insuffisant"
        FINAL_SCORE_GAUGE.set(final_score)
        SCORE_INTERPRETATION.labels(interpretation=interpretation).inc()

        # Données pour le front (scores immédiats)
        scores_payload = {
            "final_score": round(final_score, 2),
            "interpretation": interpretation,
            "details": {
                "text_analysis": {
                    "transcription": transcription,
                    "sentiment": sentiment_label,
                    "fillers": nlp_results.get("fillers_details", {}),
                },
                "audio_analysis": {
                    "volume": vol_display,
                    "tempo_bpm": f"{round(tempo_val, 1)} mots/min ({label_tempo})",
                    "pause_ratio": f"{round(features.get('pause_ratio', 0) * 100, 1)}%",
                },
                "emotion_analysis": {
                    "label": emotion_data.get("emotion", "Neutre"),
                    "confidence": f"{round(emotion_data.get('confidence', 0) * 100, 1)}%",
                },
            },
        }

        # Préparer les variables pour le rapport
        fillers = scores_payload["details"]["text_analysis"]["fillers"]
        filler_count_report = sum(fillers.values()) if isinstance(fillers, dict) else 0

    except Exception as e:
        logger.error(f"Erreur analyse stream : {e}")
        API_REQUESTS.labels(endpoint="/analyze_stream/", status="error").inc()
        # On ne peut pas renvoyer de JSON classique ici, on stream une erreur
        async def error_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

    # Générateur SSE
    async def event_stream():
        # Phase 1 : scores immédiats
        yield f"data: {json.dumps({'type': 'scores', 'data': scores_payload})}\n\n"

        # Phase 2 : rapport structuré (5 sections), ligne par ligne
        try:
            rapport_base, points_forts, axes = generate_structured_report(
                score=scores_payload["final_score"],
                interpretation=interpretation,
                sentiment_label=sentiment_label,
                emotion=emotion_data.get("emotion", "Neutre"),
                vol_raw=vol_raw,
                vol_display=vol_display,
                tempo_val=tempo_val,
                label_tempo=label_tempo,
                pause_ratio_pct=round(features.get("pause_ratio", 0) * 100, 1),
                filler_count=filler_count_report,
                fillers_dict=fillers if isinstance(fillers, dict) else {},
                transcription=transcription,
                word_count=nlp_results.get("word_count", 0),
                llm_conclusion=None,
            )

            for line in rapport_base.split("\n"):
                yield f"data: {json.dumps({'type': 'report_line', 'text': line})}\n\n"

        except Exception as e:
            logger.error(f"Erreur génération rapport stream : {e}")
            yield f"data: {json.dumps({'type': 'report_line', 'text': 'Rapport non disponible.'})}\n\n"

        # Phase 3 : conclusion LLM en streaming token par token
        try:
            # Séparateur section 6
            sep = "─" * 60
            for line in [sep, "6. CONCLUSION PERSONNALISÉE", sep]:
                yield f"data: {json.dumps({'type': 'report_line', 'text': line})}\n\n"

            has_tokens = False
            for token in generate_conclusion_stream(
                score=scores_payload["final_score"],
                interpretation=interpretation,
                emotion=emotion_data.get("emotion", "Neutre"),
                sentiment_label=sentiment_label,
                filler_count=filler_count_report,
                tempo_val=tempo_val,
                pause_ratio_pct=round(features.get("pause_ratio", 0) * 100, 1),
                word_count=nlp_results.get("word_count", 0),
                points_forts=points_forts,
                axes=axes,
            ):
                has_tokens = True
                yield f"data: {json.dumps({'type': 'llm_token', 'text': token})}\n\n"

            if not has_tokens:
                yield f"data: {json.dumps({'type': 'report_line', 'text': '(Conclusion LLM non disponible)'})}\n\n"

        except Exception as e:
            logger.warning(f"Conclusion LLM stream échouée : {e}")
            yield f"data: {json.dumps({'type': 'report_line', 'text': '(Conclusion LLM non disponible)'})}\n\n"

        # Phase 4 : analyse de pertinence question/réponse (si question d'entretien fournie)
        if interview_question.strip():
            try:
                sep = "─" * 60
                for line in [sep, "7. PERTINENCE DE LA RÉPONSE", sep]:
                    yield f"data: {json.dumps({'type': 'report_line', 'text': line})}\n\n"

                yield f"data: {json.dumps({'type': 'report_line', 'text': f'Question posée : « {interview_question.strip()} »'})}\n\n"
                yield f"data: {json.dumps({'type': 'report_line', 'text': ''})}\n\n"

                has_relevance = False
                for token in generate_relevance_stream(
                    question=interview_question.strip(),
                    transcription=transcription,
                    domain=interview_domain.strip(),
                    position=interview_position.strip(),
                ):
                    has_relevance = True
                    yield f"data: {json.dumps({'type': 'relevance_token', 'text': token})}\n\n"

                if not has_relevance:
                    yield f"data: {json.dumps({'type': 'report_line', 'text': '(Analyse de pertinence non disponible)'})}\n\n"

            except Exception as e:
                logger.warning(f"Analyse pertinence stream échouée : {e}")
                yield f"data: {json.dumps({'type': 'report_line', 'text': '(Analyse de pertinence non disponible)'})}\n\n"

        # Persistance DB (non bloquant)
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

        API_REQUESTS.labels(endpoint="/analyze_stream/", status="success").inc()

        # Phase finale : fin
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )