# 1. Utilisation d'une image stable
FROM python:3.10-slim

# 2. Variables d'environnement pour Python
# PYTHONDONTWRITEBYTECODE: Empêche la création de fichiers .pyc (gain de place)
# PYTHONUNBUFFERED: Affiche les logs en temps réel (crucial pour Docker/FastAPI)
# PYTHONPATH: Permet de trouver les modules dans /app/src et /app/api
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

# 3. Installation des dépendances système (Audio & ML)
# Ajout de ffmpeg et libsndfile1 si vous manipulez de l'audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 4. Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Pré-téléchargement des modèles (AVANT le COPY du code pour profiter du cache Docker)
# Ce layer ne se ré-exécute que si requirements.txt change — pas quand le code change
RUN mkdir -p /app/.cache/torch/hub /app/.cache/huggingface /app/.cache/whisper
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cmarkea/distilcamembert-base-sentiment', top_k=None)"
RUN python -c "import whisper; whisper.load_model('base')"

# 6. Copie du code source et des assets
COPY api/ ./api/
COPY src/ ./src/
COPY config/ ./config/
COPY database/ ./database/
COPY storage/ ./storage/

# 7. Sécurité : On évite de tourner en 'root'
RUN useradd -m interviewuser \
    && chown -R interviewuser /app
USER interviewuser

EXPOSE 8000

# 7. Lancement avec Uvicorn
# On utilise la notation api.main:app car api est un package (contient __init__.py)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]