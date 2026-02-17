# InterviewFlow AI

**InterviewFlow AI** est une application B2B complète de simulation d'entretiens, d'analyse audio/texte et de feedback soft skills, pensée pour les entreprises, écoles et coachs RH.

## Fonctionnalités principales
- **Simulation d'entretien** : Enregistrement audio via le navigateur, analyse en temps réel.
- **Analyse IA** : Extraction de métriques audio (stress, dynamisme, pauses) et NLP (sentiment, fillers, transcription).
- **Feedback Soft Skills** : Calcul automatique de stress, confiance, dynamisme.
- **Dashboard** : Visualisation des historiques, courbes, logs, et performances modèles.
- **API REST** : Exposition des modèles et analyses via FastAPI.
- **MLOps** : Intégration MLflow (tracking), Prometheus (metrics), Grafana (dashboard).
- **Tests unitaires** : Pour pipeline, modèles ML/DL, etc.

## Architecture du projet (arborescence complète)
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
│   └── main.py
├── config/
│   └── settings.yaml
├── database/
│   └── db_manager.py
├── docker-compose.yml
├── frontend/
│   ├── Dockerfile
│   ├── main.py
│   └── assets/
│       └── style.css
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
│   └── simulate_data.py
├── src/
│   ├── data_pipeline.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/
│   │   ├── dl_model.py
│   │   └── ml_model.py
│   ├── monitoring/
│   │   ├── metrics.py
│   │   └── mlflow/
│   │       ├── setup.py
│   │       └── utils.py
│   └── processors/
│       ├── audio_engine.py
│       └── nlp_engine.py
├── storage/
│   ├── models/   # (vide)
│   └── storage_manager.py
├── tests/
│   └── unit/
│       ├── test_data_pipeline.py
│       ├── test_dl_model.py
│       └── test_ml_model.py
├── venv/   # (environnement virtuel)
```
   ```bash
   uvicorn api.app:app --reload
   ```
6. **Lancer le frontend**
   ```bash
   streamlit run app/main.py
   ```

## MLOps & Monitoring
- **MLflow** :
  - Lancer le tracking server :
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
    ```
- **Prometheus** :
  - Lancer avec la config :
    ```bash
    prometheus --config.file=prometheus/prometheus_config.yaml
    ```
- **Grafana** :
  - Importer `infra/grafana/dashboards_json/dashboard.json` ou `infra/grafana/provisioning/dashboards/dashboard.yml` et connecter Prometheus comme datasource.

## Tests
```bash
python -m unittest discover -s tests/unit
```

## Points d’entrée principaux
- **API** : http://localhost:8000/docs
- **App** : http://localhost:8501
- **MLflow UI** : http://localhost:5000
- **Grafana** : http://localhost:3000
- **Prometheus** : http://localhost:9090

## Personnalisation
- Modifiez `config/settings.yaml` pour ajuster les seuils audio/NLP.
- Ajoutez vos propres modèles dans `src/models/`.
- Ajoutez des panels Grafana selon vos besoins.

## Auteurs & Licence
Projet réalisé par [Votre Nom] dans le cadre du PFE 2026. Licence MIT.
