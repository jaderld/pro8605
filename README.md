# 

**Ce projet** est une application B2B complète de simulation d'entretiens, d'analyse audio/texte et de feedback soft skills, pensée pour les entreprises, écoles et coachs RH.

## Fonctionnalités principales
- **Simulation d'entretien** : Enregistrement audio via le navigateur, analyse en temps réel.
- **Analyse IA** : Extraction de métriques audio (stress, dynamisme, pauses) et NLP (sentiment, fillers, transcription).
- **Feedback Soft Skills** : Calcul automatique de stress, confiance, dynamisme.
- **Dashboard** : Visualisation des historiques, courbes, logs, et performances modèles.
- **API REST** : Exposition des modèles et analyses via FastAPI.
- **MLOps** : Intégration MLflow (tracking), Prometheus (metrics), Grafana (dashboard).
- **Tests unitaires** : Pour pipeline, modèles ML/DL, etc.


# Projet PFE PRO 8605

Ce projet est une plateforme complète d’analyse d’entretiens, d’audio et de texte, avec scoring automatique, monitoring, et visualisation. Il s’adresse aux besoins de simulation, d’évaluation, et de feedback pour des applications RH, pédagogiques ou de recherche.

---

## Fonctionnalités

- **Simulation d’entretien** : Enregistrement audio, analyse en temps réel.
- **Extraction de métriques** : Audio (volume, tempo, pauses, stress), NLP (sentiment, fillers, transcription).
- **Scoring automatique** : Modèles ML/DL pour score global, émotion, soft skills.
- **Historique & base de données** : Sauvegarde des sessions, accès aux résultats.
- **API REST** : Exposition des analyses, endpoints d’entraînement, healthcheck.
- **Monitoring MLOps** : MLflow (tracking runs, modèles, artefacts), Prometheus (metrics), Grafana (dashboards).
- **Tests unitaires** : Validation pipeline, modèles, intégration.
- **Personnalisation** : Paramètres via YAML, ajout de modèles, dashboards.

---

## Architecture du projet

```
pro8605/
├── .dockerignore
├── .env
├── .git/
├── .github/
├── .gitignore
├── api/
│   ├── Dockerfile
│   ├── __init__.py
│   ├── main.py
│   ├── static/
│   │   ├── index.html
│   │   ├── script.js
│   │   └── style.css
│   └── __pycache__/
├── config/
│   └── settings.yaml
├── database/
│   └── db_manager.py
├── docker-compose.yml
├── infra/
│   ├── grafana/
│   │   ├── dashboards_json/
│   │   │   └── dashboard.json
│   │   ├── grafana.ini
│   │   └── provisioning/
│   │       ├── dashboards/
│   │       │   └── dashboard.yml
│   │       └── datasources/
│   │           └── datasource.yml
│   └── prometheus/
│       └── prometheus.yaml
├── Makefile
├── README.md
├── requirements.txt
├── scripts/
│   ├── simulate_data.py
│   └── test_nlp_bypass.py
├── src/
│   ├── data_pipeline.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/
│   │   ├── dl_model.py
│   │   ├── ml_model.py
│   │   └── __pycache__/
│   ├── monitoring/
│   │   ├── metrics.py
│   │   ├── mlflow/
│   │   │   ├── setup.py
│   │   │   └── utils.py
│   │   └── __pycache__/
│   └── processors/
│       ├── audio_engine.py
│       ├── nlp_engine.py
│       └── __pycache__/
├── storage/
│   ├── fake_sessions.csv
│   ├── storage_manager.py
│   ├── mlflow/
│   │   ├── artifacts/
│   │   └── mlflow.db
│   └── models/
├── tests/
│   └── unit/
│       ├── test_data_pipeline.py
│       ├── test_dl_model.py
│       └── test_ml_model.py
├── venv/
```

---

## Pipeline d’analyse

1. **Upload audio** (API ou frontend)
2. **AudioEngine** : Extraction des features audio
3. **DL Model** : Transcription (Whisper), prédiction émotion/stress (PyTorch)
4. **NLP Engine** : Analyse du texte (sentiment, fillers)
5. **ML Model** : Scoring global (LogisticRegression/RandomForest)
6. **Sauvegarde session** : Enregistrement des résultats en base SQLite (via `DBManager`)
7. **Monitoring** : Export des métriques Prometheus, tracking MLflow
8. **Visualisation** : Dashboards Grafana, historique sessions

---

## Modules principaux

- **api/** : Backend FastAPI, endpoints d’analyse, entraînement, monitoring, frontend statique
- **src/** : Logique métier (modèles ML/DL, pipeline, NLP/audio processors, monitoring)
- **database/** : Gestion base SQLite, sauvegarde et récupération des sessions
- **infra/** : Config Grafana/Prometheus
- **storage/** : Données simulées, modèles sauvegardés, artefacts MLflow
- **config/** : Paramètres YAML (seuils, fillers, etc)
- **scripts/** : Génération de données fictives, tests NLP
- **tests/** : Tests unitaires pipeline, modèles

---

## Entraînement & Évaluation

- **ML Model** :
  - Modèle de scoring (LogisticRegression/RandomForest)
  - Méthodes : train, predict, evaluate, save, load
  - Entraînement via endpoint ou script, données simulées (`fake_sessions.csv`)
- **DL Model** :
  - Réseau PyTorch pour émotion/stress
  - Transcription audio via Whisper
  - Méthodes d’entraînement, sauvegarde
- **NLP Engine** :
  - Analyse sentiment, fillers, transcription
- **DBManager** :
  - Sauvegarde des sessions, récupération historique

---

## Monitoring & MLOps

- **MLflow** : Tracking runs, modèles, artefacts (UI : http://localhost:5000)
- **Prometheus** : Scraping métriques exposées par l’API (http://localhost:9090)
- **Grafana** : Dashboards pour visualiser métriques, historiques, performances modèles (http://localhost:3000)
- **Logs** : Logging standard Python, logs en temps réel (API, scripts, modèles)

---

## Tests

Lancer tous les tests unitaires :
```bash
python -m unittest discover -s tests/unit
```

---

## Utilisation rapide

1. **Build & lancement des services**
   ```bash
   docker-compose up --build
   ```
2. **Accéder à l’API** : http://localhost:8000/docs
3. **Accéder à MLflow** : http://localhost:5000
4. **Accéder à Grafana** : http://localhost:3000
5. **Accéder à Prometheus** : http://localhost:9090

---

## Personnalisation

- Modifiez `config/settings.yaml` pour ajuster les seuils audio/NLP
- Ajoutez vos propres modèles dans `src/models/`
- Ajoutez des panels Grafana selon vos besoins
- Modifiez la base de données ou le pipeline selon vos besoins

---

