# --- VARIABLES ---
DOCKER_COMPOSE = docker-compose
PYTHON = python3

# --- COULEURS ---
BLUE = \033[1;34m
GREEN = \033[1;32m
NC = \033[0m

.PHONY: help build up down restart logs ps test-nlp clean

help:
	@echo "$(BLUE)InterviewFlow AI - Commandes disponibles :$(NC)"
	@echo "$(GREEN)  make build$(NC)        : Construit les images Docker"
	@echo "$(GREEN)  make up$(NC)           : Lance tous les services (API, Front, MLflow, etc.)"
	@echo "$(GREEN)  make down$(NC)         : Arrête tous les services"
	@echo "$(GREEN)  make restart$(NC)      : Redémarre les services"
	@echo "$(GREEN)  make logs$(NC)         : Affiche les logs en temps réel"
	@echo "$(GREEN)  make test-nlp$(NC)     : Teste l'analyse sémantique avec des textes bruts"
	@echo "$(GREEN)  make simulate$(NC)     : Génère des données fictives pour Grafana"
	@echo "$(GREEN)  make clean$(NC)        : Supprime les fichiers temporaires et les caches"

# --- DOCKER OPS ---
build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) up -d
	@echo "$(BLUE)Services lancés :$(NC)"
	@echo "  - API : http://localhost:8000/docs"
	@echo "  - App : http://localhost:8501"
	@echo "  - MLflow : http://localhost:5000"
	@echo "  - Grafana : http://localhost:3000"

down:
	$(DOCKER_COMPOSE) down

restart:
	$(DOCKER_COMPOSE) restart

logs:
	$(DOCKER_COMPOSE) logs -f

ps:
	$(DOCKER_COMPOSE) ps

# --- TESTS & SIMULATION ---

# Cette commande permet de tester ton module NLP directement
test-nlp:
	@echo "$(BLUE)Test de l'analyse sémantique (Textes avec tics)...$(NC)"
	$(PYTHON) scripts/simulate_data.py --mode nlp-test

simulate:
	@echo "$(BLUE)Génération de données pour Grafana...$(NC)"
	$(PYTHON) scripts/simulate_data.py --mode metrics

# --- NETTOYAGE ---
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache