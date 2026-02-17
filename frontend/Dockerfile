# 1. Image de base légère
FROM python:3.10-slim

# 2. Configuration environnement
# PYTHONPATH=/app permet d'importer 'src' ou 'frontend' sans erreur
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# 3. Dépendances système (Streamlit en a parfois besoin pour le rendu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copie du code (Frontend + Source commune)
# On copie src car le front peut en avoir besoin pour la logique métier
COPY frontend/ ./frontend/
COPY src/ ./src/
COPY config/ ./config/

# 6. Exposition du port Streamlit par défaut
EXPOSE 8501

# 7. Configuration de Streamlit pour Docker
# --server.address=0.0.0.0 est indispensable pour l'accès externe
# --server.port=8501 force le port au cas où la config change
CMD ["streamlit", "run", "frontend/main.py", "--server.port=8501", "--server.address=0.0.0.0"]