# PRO8605 — Plateforme d'Analyse Soft Skills & Simulateur d'Entretien

> Projet de Fin d'Études — Application B2B d'analyse comportementale audio/texte en temps réel, avec scoring IA, monitoring MLOps et pipeline complet de Machine Learning.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture](#2-architecture)
3. [Pipeline d'analyse temps réel](#3-pipeline-danalyse-temps-réel)
4. [Modèles — Fonctionnement & Entraînement](#4-modèles--fonctionnement--entraînement)
   - 4.1 [AudioEngine — Extraction de features](#41-audioengine--extraction-de-features)
   - 4.2 [Modèle DL — SimpleAudioNet (PyTorch)](#42-modèle-dl--simpleaudionet-pytorch)
   - 4.3 [Transcription — Whisper (OpenAI)](#43-transcription--whisper-openai)
   - 4.4 [NLPEngine — DistilCamemBERT](#44-nlpengine--distilcamembert)
   - 4.5 [Modèle ML — Random Forest Regressor](#45-modèle-ml--random-forest-regressor)
   - 4.6 [Rapport structuré — Générateur basé sur règles](#46-rapport-structuré--générateur-basé-sur-règles)
   - 4.7 [Endpoints d'entraînement à la demande](#47-endpoints-dentraînement-à-la-demande)
5. [Métriques évaluées](#5-métriques-évaluées)
6. [MLOps — Tracking & Monitoring](#6-mlops--tracking--monitoring)
7. [Données d'entraînement](#7-données-dentraînement)
8. [Installation & Lancement](#8-installation--lancement)
9. [Tests unitaires](#9-tests-unitaires)
10. [Structure du projet](#10-structure-du-projet)
11. [Limites & Roadmap](#11-limites--roadmap)

---

## 1. Vue d'ensemble

PRO8605 est une plateforme web full-stack permettant à un candidat (ou formateur RH) d'enregistrer une prise de parole depuis le navigateur et d'obtenir en retour une **analyse automatique complète** de ses soft skills :

| Dimension | Ce qui est mesuré |
|---|---|
| **Voix physique** | Volume RMS, débit (BPM), ratio de silences |
| **Émotion** | Classification binaire Calme / Stressé (PyTorch) |
| **Discours** | Transcription automatique (Whisper), sentiment (DistilCamemBERT), tics de langage |
| **Score global** | Note /100 calculée par Random Forest + pénalités |
| **Rapport détaillé** | Texte structuré généré par règles métier RH (5 sections) |

---

## 2. Architecture

```mermaid
flowchart TD
  A["Interface Web\n(HTML / JS)"] -->|"POST /analyze_file/"| B["API FastAPI\n(api/main.py)"]
  B --> C["AudioEngine\nlibrosa + Silero-VAD"]
  B --> D["Whisper\nOpenAI base model"]
  B --> E["NLPEngine\nDistilCamemBERT + Regex"]
  B --> F["SimpleAudioNet\nPyTorch - 2 classes"]
  B --> G["ScoringModel\nRandom Forest Regressor"]
  B --> H["ReportGenerator\nRegles metier RH"]
  C --> F
  D --> E
  E --> G
  F --> G
  B -->|"Sauvegarde session"| I[("PostgreSQL\nfallback SQLite")]
  B -->|"Exposition metriques"| J["Prometheus\n:9090/metrics"]
  J --> K["Grafana\n:3000"]
  B -->|"Log runs ML"| L["MLflow\n:5000"]
```

### Services Docker

| Service | Port | Rôle |
|---|---|---|
| `api` | 8000 | Backend FastAPI + Frontend statique |
| `mlflow` | 5000 | Tracking des expériences ML |
| `postgres` | 5432 | Base de données sessions |
| `prometheus` | 9090 | Collecte des métriques temps réel |
| `grafana` | 3000 | Dashboard de visualisation |

---

## 3. Pipeline d'analyse temps réel

```
① Réception du fichier audio (WAV via MediaRecorder)
② AudioEngine.process_signal()
   → librosa : RMS volume, tempo (beat_track), ZCR, spectral_centroid
   → Silero-VAD : détection des segments de parole → pause_ratio
   → dl_input_vector = [rms, zcr, spec_centroid/1000, tempo/200, pause_ratio]
③ InterviewModel.transcribe_audio()     [Whisper base, fr]
④ InterviewModel.predict_emotion()      [SimpleAudioNet]
⑤ NLPEngine.analyze_text()              [DistilCamemBERT + Regex]
⑥ ScoringModel.predict_score()          [Random Forest]
⑦ Calcul du score final avec pénalités  [tics x10 + sentiment_penalty]
⑧ Génération rapport structuré          [règles métier RH, 5 sections]
⑨ Réponse JSON → affichage UI
```

---

## 4. Modèles — Fonctionnement & Entraînement

### 4.1 AudioEngine

Features extraites via `librosa` et `Silero-VAD` :

| Feature | Méthode | Normalisation |
|---|---|---|
| `volume` | RMS moyen | brut |
| `tempo` | beat_track | /200.0 pour DL |
| `zcr` | zero_crossing_rate | brut |
| `spectral_centroid` | spectral_centroid | /1000.0 pour DL |
| `pause_ratio` | Silero-VAD | [0, 1] |

Vecteur d'entrée DL :
```python
dl_input_vector = [rms, zcr, spectral_centroid/1000.0, tempo/200.0, pause_ratio]
```

### 4.2 Modèle DL — SimpleAudioNet (PyTorch)

Architecture :
```
Input (5) → Linear(5→64) → ReLU → Dropout(0.2)
          → Linear(64→32) → ReLU
          → Linear(32→2)  → [Softmax à l'inférence]
```

Hyperparamètres : `epochs=50`, `batch_size=16`, `Adam(lr=0.001)`, `CrossEntropyLoss`, `Dropout=0.2`

Évaluation (loggée dans MLflow) :
- `train_loss`, `test_accuracy`, `test_f1` par epoch (courbes de convergence)
- En fin d'entraînement : `precision`, `recall`, matrice de confusion (JSON), rapport de classification (texte)

### 4.3 Transcription — Whisper

Whisper `base` (74M params), `language="fr"`, avec prompt d'amorçage ciblant les hésitations françaises :
```python
initial_prompt = "C'est un entretien d'embauche. Le candidat hésite souvent, il dit euh, bah, voilà, du coup."
```

### 4.4 NLPEngine — DistilCamemBERT

- **Modèle** : `cmarkea/distilcamembert-base-sentiment` — distillation de CamemBERT, nativement français
- **Tâche** : classification 5 classes (1 à 5 étoiles) → score continu [-1, +1] par moyenne pondérée

```
score = sum( (i-3)/2 * p_i  for i in [1..5] )
```

- **Tics de langage** : détection Regex `\b mot \b` sur liste configurable (`config/settings.yaml`)
- **Truncation** : 2048 caractères (≈ 512 tokens, limite du modèle)

### 4.5 Modèle ML — Random Forest Regressor

Features : `['volume', 'tempo', 'pause_ratio', 'sentiment', 'filler_count', 'stress_level']`

Hyperparamètres : `n_estimators=100`, `max_depth=10`, `test_size=20%`, `random_state=42`

Évaluation (loggée dans MLflow) :
- Courbe de convergence : MAE + R² aux checkpoints `[1, 5, 10, 25, 50, 75, 100]` arbres
- **K-Fold Cross-Validation (5 folds)** sur l'ensemble complet : `cv_mae_mean ± std`, `cv_r2_mean ± std`
- Importance des 6 features
- Artefact JSON d'évaluation complet (`evaluation_summary.json`)

### 4.6 Rapport structuré — Générateur basé sur règles

`src/processors/report_generator.py` — 5 sections déterministes :

| Section | Contenu |
|---|---|
| 1. Résumé global | Performance globale selon seuils de score |
| 2. Analyse des scores | Volume, tempo, pauses, tics — comparaison aux seuils RH |
| 3. Analyse de la transcription | Nombre de mots, richesse du discours, détail des tics |
| 4. Feedback personnalisé | Points forts + axes d'amélioration ciblés |
| 5. Conseils pratiques | Recommandations concrètes (méthode STAR, respiration, débit...) |

### 4.7 Endpoints d'entraînement

| Route | Méthode | Effet |
|---|---|---|
| `/train/dl` | POST | Entraîne SimpleAudioNet, sauvegarde `emotion_net.pth`, log MLflow |
| `/train/ml` | POST | Entraîne Random Forest, sauvegarde `scoring_rf.joblib`, log MLflow |

---

## 5. Métriques évaluées

| Modèle | Métrique | Valeur typique |
|---|---|---|
| SimpleAudioNet | Accuracy | ~70–80% |
| SimpleAudioNet | F1-score (weighted) | ~0.70–0.78 |
| SimpleAudioNet | Precision / Recall | loggés par epoch |
| Random Forest | MAE (holdout) | ~6–10 pts |
| Random Forest | R2 (holdout) | ~0.70–0.85 |
| Random Forest | MAE (5-fold CV) | stabilisé ± std |

---

## 6. MLOps — Tracking & Monitoring

### MLflow — Expériences d'entraînement

- URL : http://localhost:5000
- Expériences : `Audionet_DL`, `Final_Scoring_ML`
- Artefacts loggés : modèles `.pth`/`.joblib`, rapports de classification, matrices de confusion, résumés d'évaluation JSON

### Prometheus — Métriques d'inférence temps réel

- URL : http://localhost:9090
- Scrape toutes les 10s depuis `api:8000/metrics`
- Métriques par étape du pipeline :

| Etape | Métriques |
|---|---|
| Audio | `audio_duration_seconds`, `audio_pause_ratio`, `audio_tempo_bpm`, `audio_volume_rms` |
| Transcription | `dl_transcription_time_seconds`, `transcription_word_count` |
| NLP | `nlp_filler_words_total`, `nlp_filler_words_per_analysis`, `nlp_sentiment_score` |
| Emotion | `audio_stress_level`, `model_inference_time_seconds`, `model_prediction_confidence` |
| Scoring | `interview_final_score`, `interview_base_score`, `interview_score_penalty`, `interview_score_interpretation_total` |
| API | `api_requests_total` |

### Grafana — Dashboard

- URL : http://localhost:3000
- 18 panneaux organisés en 5 sections (pipeline stages)
- Auto-provisionné au démarrage

---

## 7. Données d'entraînement

`storage/fake_sessions.csv` — 2000 sessions synthétiques, 12% de bruit de labels

Distribution par classe :
- `label=1` (Stressé) : `volume~N(0.05, 0.03)`, `bpm~N(130, 25)`, `pause_ratio~N(0.35, 0.15)`
- `label=0` (Calme)   : `volume~N(0.07, 0.04)`, `bpm~N(110, 20)`, `pause_ratio~N(0.18, 0.10)`

Formule du score cible :
```python
target_score = 85 - pause_ratio*60 - filler_count*3 + sentiment*10 - label*5 + bruit(+-12)
```

---

## 8. Installation & Lancement

### Docker (recommandé)

```bash
docker-compose up --build -d
```

Les modèles HuggingFace (DistilCamemBERT, ~260 Mo) et Whisper (~150 Mo) sont téléchargés au build
et mis en cache dans des volumes Docker persistants (`hf_cache`, `whisper_cache`, `torch_cache`).

### Local (développement)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Générer les données d'entraînement
python scripts/simulate_data.py

# Lancer l'API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Entraînement des modèles

```bash
# Via API (avec données fake_sessions.csv)
curl -X POST http://localhost:8000/train/dl
curl -X POST http://localhost:8000/train/ml

# Via Makefile
make train-dl
make train-ml
```

---

## 9. Tests unitaires

```bash
python -m unittest discover -s tests/unit

# Tests individuels
python -m unittest tests.unit.test_ml_model
python -m unittest tests.unit.test_dl_model
python -m unittest tests.unit.test_data_pipeline

# Test NLP + ML (sans audio, sans Docker)
make test-nlp
```

---

## 10. Structure du projet

```
pro8605/
├── api/
│   ├── Dockerfile
│   ├── main.py
│   └── static/                 (index.html, script.js, style.css)
├── config/
│   └── settings.yaml           (sample_rate, thresholds, fillers)
├── database/
│   └── db_manager.py           (SQLite / PostgreSQL dual-backend)
├── docker-compose.yml
├── infra/
│   ├── grafana/                (dashboard JSON, provisioning)
│   └── prometheus/             (prometheus.yaml)
├── Makefile
├── requirements.txt
├── scripts/
│   ├── init_postgres.py
│   ├── simulate_data.py
│   └── test_nlp_bypass.py
├── src/
│   ├── data_pipeline.py
│   ├── models/
│   │   ├── dl_model.py         (Whisper + SimpleAudioNet)
│   │   └── ml_model.py         (Random Forest Regressor)
│   ├── monitoring/
│   │   ├── metrics.py          (Prometheus — 18 metriques)
│   │   └── mlflow/
│   │       ├── setup.py        (init_mlflow)
│   │       └── utils.py        (log_params, log_step_metrics, ...)
│   └── processors/
│       ├── audio_engine.py     (librosa + Silero-VAD)
│       ├── nlp_engine.py       (DistilCamemBERT + Regex)
│       └── report_generator.py (rapport structuré 5 sections)
├── storage/
│   ├── fake_sessions.csv
│   └── models/
│       ├── emotion_net.pth
│       └── scoring_rf.joblib
└── tests/unit/
    ├── test_data_pipeline.py
    ├── test_dl_model.py
    └── test_ml_model.py
```

---

## 11. Limites & Roadmap

| Aspect | Limite actuelle |
|---|---|
| Données | Dataset synthétique — distribution gaussienne, biais possible en conditions réelles |
| SimpleAudioNet | MLP sur 5 features tabulaires — pas un modèle audio profond (pas de spectrogramme) |
| Whisper | Lent sur CPU (~10–30s/min audio) — utiliser large-v3 pour plus de précision |
| DistilCamemBERT | Entrainé sur avis produits (1–5 etoiles), pas sur discours d'entretien |
| Auth | Aucune authentification — usage académique uniquement |
| Cross-validation DL | Absente pour le modèle PyTorch (présente uniquement pour RF) |

Roadmap :
- JWT auth + rôles (candidat / recruteur)
- Modèle audio profond sur spectrogrammes (CNN sur MFCC ou wav2vec2)
- Fine-tuning CamemBERT sur corpus d'entretiens
- Streaming audio long (> 5 minutes)
- Alerting Grafana (seuils de score)
- API LLM externe (GPT-4o) pour rapport enrichi
