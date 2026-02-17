import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from src.processors.audio_engine import AudioEngine
from src.processors.nlp_engine import NLPEngine
from src.evaluation.metrics import compute_soft_skills
from src.models.ml_model import MLModel
from src.models.dl_model import DLModel
from src.data_pipeline import DataPipeline
from storage.storage_manager import StorageManager


app = FastAPI(title="InterviewFlow AI API", description="API for Interview Simulation, Analysis, and ML/DL Models.", version="1.0")
audio_engine = AudioEngine()
nlp_engine = NLPEngine()
ml_model = MLModel()
dl_model = None  # Will be loaded on demand
pipeline = DataPipeline(data_path='storage/fake_sessions.csv')
storage = StorageManager()


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok"}


@app.post("/analyze_file/", tags=["Analysis"])
async def analyze_file(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        audio_metrics = audio_engine.process_signal(temp_path)
        nlp_metrics = nlp_engine.process_text(temp_path)
        soft_skills = compute_soft_skills(audio_metrics, nlp_metrics)
        result = {**audio_metrics, **nlp_metrics, **soft_skills}
        # Save audio file
        storage.save_audio(temp_path, file.filename)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except Exception:
            pass

# ML Model endpoints
@app.post("/ml/train/", tags=["ML Model"])
def train_ml_model():
    try:
        df = pipeline.extract()
        df = pipeline.transform(df)
        X_train, X_test, y_train, y_test = pipeline.split(df, 'label')
        ml_model.train(X_train, y_train)
        acc = ml_model.evaluate(X_test, y_test)
        ml_model.save('storage/models/ml_model.pkl')
        return {"status": "trained", "accuracy": acc}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ml/predict/", tags=["ML Model"])
def predict_ml_model(features: dict):
    try:
        import numpy as np
        X = np.array([list(features.values())])
        preds = ml_model.predict(X)
        return {"prediction": int(preds[0])}
    except Exception as e:
        return {"error": str(e)}

# DL Model endpoints
@app.post("/dl/train/", tags=["DL Model"])
def train_dl_model():
    global dl_model
    try:
        df = pipeline.extract()
        df = pipeline.transform(df)
        X_train, X_test, y_train, y_test = pipeline.split(df, 'label')
        input_dim = X_train.shape[1]
        num_classes = len(set(y_train))
        dl_model = DLModel(input_dim, num_classes)
        dl_model.train(X_train, y_train, epochs=5)
        dl_model.save('storage/models/dl_model.pt')
        return {"status": "trained"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/dl/predict/", tags=["DL Model"])
def predict_dl_model(features: dict):
    global dl_model
    try:
        import numpy as np
        if dl_model is None:
            dl_model = DLModel(len(features), 2)
            dl_model.load('storage/models/dl_model.pt')
        X = np.array([list(features.values())])
        preds = dl_model.predict(X)
        return {"prediction": int(preds[0])}
    except Exception as e:
        return {"error": str(e)}

# Data management endpoints
@app.get("/data/sessions/", tags=["Data"])
def get_sessions():
    try:
        import pandas as pd
        df = pd.read_csv('storage/fake_sessions.csv')
        return df.to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}
