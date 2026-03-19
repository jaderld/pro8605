# --- VARIABLES ---
DOCKER_COMPOSE = docker-compose
PYTHON = python

# --- COULEURS ---
BLUE = \033[1;34m
GREEN = \033[1;32m
NC = \033[0m

.PHONY: help build up down restart logs ps train-dl train-ml test-nlp simulate clean

help:
	@echo "$(BLUE)PRO8605 - Commandes disponibles :$(NC)"
	@echo "$(GREEN)  make build$(NC)        : Construit les images Docker"
	@echo "$(GREEN)  make up$(NC)           : Lance tous les services (API, Front, MLflow, etc.)"
	@echo "$(GREEN)  make down$(NC)         : Arrête tous les services"
	@echo "$(GREEN)  make restart$(NC)      : Redémarre les services"
	@echo "$(GREEN)  make logs$(NC)         : Affiche les logs en temps réel"
	@echo "$(GREEN)  make train-dl$(NC)     : Entraîne le modèle DL (SimpleAudioNet)"
	@echo "$(GREEN)  make train-ml$(NC)     : Entraîne le modèle ML (Random Forest)"
	@echo "$(GREEN)  make test-nlp$(NC)     : Teste l'analyse sémantique avec des textes bruts"
	@echo "$(GREEN)  make simulate$(NC)     : Génère des données fictives pour Grafana"
	@echo "$(GREEN)  make clean$(NC)        : Supprime les fichiers temporaires et les caches"

# --- DOCKER OPS ---
build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) up -d
	@echo "$(BLUE)Services lancés :$(NC)"
	@echo "  - API + Interface : http://localhost:8000"
	@echo "  - Swagger / Docs  : http://localhost:8000/docs"
	@echo "  - MLflow          : http://localhost:5000"
	@echo "  - Prometheus      : http://localhost:9090"
	@echo "  - Grafana         : http://localhost:3000"

down:
	$(DOCKER_COMPOSE) down

restart:
	$(DOCKER_COMPOSE) restart

logs:
	$(DOCKER_COMPOSE) logs -f

ps:
	$(DOCKER_COMPOSE) ps

# --- ENTRAÎNEMENT ---

train-dl:
	@echo "$(BLUE)Entraînement du modèle DL (SimpleAudioNet)...$(NC)"
	curl -s -X POST http://localhost:8000/train/dl | $(PYTHON) -m json.tool

train-ml:
	@echo "$(BLUE)Entraînement du modèle ML (Random Forest)...$(NC)"
	curl -s -X POST http://localhost:8000/train/ml | $(PYTHON) -m json.tool

# --- TESTS & SIMULATION ---

# Lance le script d'intégration NLP (textes bruts sans audio)
test-nlp:
	@echo "$(BLUE)Test de l'analyse sémantique (Textes avec tics)...$(NC)"
	$(PYTHON) scripts/test_nlp_bypass.py

simulate:
	@echo "$(BLUE)Génération de données pour Grafana...$(NC)"
	$(PYTHON) scripts/simulate_data.py

# --- NETTOYAGE ---
clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	$(PYTHON) -c "import shutil, pathlib; shutil.rmtree('.pytest_cache', ignore_errors=True)"