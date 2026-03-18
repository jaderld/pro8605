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

## Migration vers PostgreSQL (préparation)

Le projet utilise actuellement SQLite pour la sauvegarde des sessions. Pour une utilisation en production ou multi-utilisateur, il est recommandé de migrer vers PostgreSQL.

**Étapes prévues :**
- Adapter le fichier `database/db_manager.py` pour supporter PostgreSQL (utilisation de SQLAlchemy ou psycopg2).
- Ajouter la configuration PostgreSQL dans `config/settings.yaml` et dans `docker-compose.yml`.
- Mettre à jour la documentation d’installation.

> La migration n’est pas encore effective mais la structure du code est pensée pour faciliter cette évolution.

### Utilisation PostgreSQL

1. Lancer le service PostgreSQL avec Docker Compose :
  ```bash
  docker-compose up -d postgres
  ```
2. Initialiser la table sessions :
  ```bash
  python scripts/init_postgres.py
  ```
3. Modifier la configuration pour utiliser PostgreSQL dans le code :
  ```python
  from database.db_manager import DBManager
  import yaml
  with open('config/settings.yaml', 'r') as f:
     cfg = yaml.safe_load(f)
  db = DBManager(use_postgres=True, pg_config=cfg['postgres'])
  ```
4. Tester la sauvegarde et la récupération des sessions.

> Pour revenir à SQLite, il suffit de passer `use_postgres=False`.

## Captures d'écran

> Ajoutez ici des captures d’écran de l’interface web, du dashboard Grafana, de MLflow, etc.

| Interface Web | Dashboard Grafana | MLflow UI |
|:-------------:|:----------------:|:---------:|
| ![web](screenshots/web.png) | ![grafana](screenshots/grafana.png) | ![mlflow](screenshots/mlflow.png) |

## Démo vidéo & Valorisation

- [Lien vers la vidéo de démonstration](#) <!-- À compléter après enregistrement -->
- [Post LinkedIn valorisant le projet](#) <!-- À compléter après publication -->

## Perspectives & Roadmap

### Ce qu’il reste à faire :

- **Données** : Récupérer un jeu de données audio plus complet et traiter des audios plus longs.
- **Stockage** : Migrer la base de données vers PostgreSQL pour une meilleure scalabilité.
- **Dashboards** : Finaliser la configuration de Prometheus et Grafana, enrichir les dashboards.
- **CI/CD** : Corriger et implémenter l’automatisation complète des tests et du déploiement.
- **Interface** : Améliorer l’interface web (ergonomie, feedback utilisateur, responsive, accessibilité).
- **Produit** : Intégrer un LLM pour fournir un rapport texte complet après chaque analyse.
- **Sécurité** : Ajouter une authentification et une gestion des accès.
- **Documentation** : Ajouter des exemples d’utilisation, des tutoriels, et enrichir la FAQ.

### Bonus valorisation
- Réaliser un post LinkedIn valorisant le projet, avec une courte vidéo de démonstration (lien à ajouter ci-dessous).

## Limites et analyse critique

- **Qualité des données** : Le jeu de données utilisé est simulé et limité en diversité. Les résultats peuvent être biaisés et ne pas généraliser à des cas réels variés.
- **Biais des modèles** : Les modèles ML/DL sont entraînés sur des données synthétiques, ce qui peut introduire des biais et limiter la robustesse.
- **Temps d'inférence** : L'analyse audio et la transcription peuvent prendre plusieurs secondes selon la longueur de l'audio et la charge serveur.
- **Gestion des longs audios** : Le traitement d'audios longs (>3 min) n'est pas encore optimisé.
- **Stockage** : Utilisation actuelle de SQLite, non adaptée à la montée en charge ou à un usage multi-utilisateur. Migration PostgreSQL prévue.
- **Dashboards** : Les dashboards Grafana sont en cours de configuration et peuvent être enrichis.
- **CI/CD** : L'automatisation des tests unitaires est en place (GitHub Actions), mais le déploiement continu reste à implémenter.
- **Interface web** : L'UI est fonctionnelle mais perfectible (ergonomie, accessibilité, responsive design).
- **LLM** : L'intégration d'un LLM pour la génération de rapports texte détaillés est en perspective.
- **Sécurité** : Les aspects sécurité (authentification, gestion des accès) ne sont pas encore traités.

## Schéma d'architecture

```mermaid
flowchart TD
  A[Interface Web (HTML/JS)] -->|Upload audio| B(API FastAPI)
  B -->|Extraction features| C(AudioEngine)
  B -->|Transcription| D(Whisper)
  B -->|Analyse NLP| E(NLPEngine)
  B -->|Prédiction émotion| F(DL Model)
  B -->|Scoring global| G(ML Model)
  B -->|Sauvegarde| H[(Base SQLite)]
  B -->|Export métriques| I(Prometheus)
  I --> J[Grafana]
  B -->|Tracking runs| K(MLflow)
  K --> L[UI MLflow]
  J --> M[Dashboard Grafana]
```


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

## Personnalisation & extension

- **Configuration** : Modifiez `config/settings.yaml` pour ajuster les seuils audio/NLP, la liste des fillers, etc.
- **Ajout de modèles** : Placez vos modèles ML/DL dans `src/models/` et adaptez les endpoints FastAPI si besoin.
- **Dashboards Grafana** : Ajoutez ou modifiez les panels dans `infra/grafana/dashboards_json/dashboard.json` et la configuration dans `infra/grafana/provisioning/`.
- **Base de données** : Modifiez `database/db_manager.py` pour changer de backend (ex : PostgreSQL).
- **Pipeline** : Ajoutez de nouveaux extracteurs de features ou étapes de traitement dans `src/processors/`.
- **Tests** : Ajoutez vos propres tests unitaires dans `tests/unit/`.
- **Frontend** : Personnalisez l’interface web dans `api/static/` (HTML, JS, CSS).

