# 1. Utilisation d'une image stable
FROM python:3.10-slim

# 2. Variables d'environnement pour Python
# PYTHONDONTWRITEBYTECODE: Empêche la création de fichiers .pyc (gain de place)
# PYTHONUNBUFFERED: Affiche les logs en temps réel (crucial pour Docker/FastAPI)
# PYTHONPATH: Permet de trouver les modules dans /app/src et /app/api
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

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

# 5. Copie du code source et des assets
# Note : On copie tout selon l'arborescence définie dans le docker-compose
COPY api/ ./api/
COPY src/ ./src/
COPY config/ ./config/
COPY database/ ./database/
COPY storage/ ./storage/

# 6. Sécurité : On évite de tourner en 'root' (optionnel mais recommandé)
RUN useradd -m interviewuser && chown -R interviewuser /app
USER interviewuser

EXPOSE 8000

# 7. Lancement avec Uvicorn
# On utilise la notation api.main:app car api est un package (contient __init__.py)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]