import os
import pickle
import joblib
from typing import Literal, Optional
from difflib import SequenceMatcher

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat

MODEL_PATH = os.getenv("MODEL_PATH", "../model/diagnosis_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "../model/scaler.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "../model/encoder.pkl")

NUMERICAL_COLS = [
    "Age",
    "Heart_Rate_bpm",
    "Body_Temperature_C",
    "Oxygen_Saturation_%",
    "Systolic",
    "Diastolic",
]

CATEGORICAL_COLS = [
    "Gender",
    "Symptom_1",
    "Symptom_2",
    "Symptom_3",
]

VALID_SYMPTOMS = [
    "Fatigue",
    "Sore throat",
    "Body ache",
    "Shortness of breath",
    "Runny nose",
    "Headache",
    "Cough",
    "Fever"
]

DIAGNOSIS_ENCODING = {
    'Healthy': 0,
    'Flu': 1,
    'Bronchitis':  2,
    'Cold':  3,
    'Pneumonia': 4,
}

REVERSE_DIAGNOSIS = {v: k for k, v in DIAGNOSIS_ENCODING.items()}

class PatientInput(BaseModel):
    Age: conint(ge=18, le=79) = Field(..., description="Patient age")
    Gender: Literal["Male", "Female"]
    Symptom_1: Literal["Fatigue", "Sore throat", "Body ache", "Shortness of breath", 
                    "Runny nose", "Headache", "Cough", "Fever"]
    Symptom_2: Literal["Fatigue", "Sore throat", "Body ache", "Shortness of breath", 
                    "Runny nose", "Headache", "Cough", "Fever"]
    Symptom_3: Literal["Fatigue", "Sore throat", "Body ache", "Shortness of breath", 
                    "Runny nose", "Headache", "Cough", "Fever"]
    Heart_Rate_bpm: conint(ge=60, le=120) = Field(..., description="Heart rate in BPM")
    Body_Temperature_C: confloat(ge=35.5, le=40.0) = Field(..., description="Body temperature in Celsius")
    Oxygen_Saturation_percent: conint(ge=90, le=99) = Field(..., alias="Oxygen_Saturation_%", description="Oxygen saturation percentage")
    Systolic: conint(ge=90, le=180) = Field(..., description="Systolic blood pressure")
    Diastolic: conint(ge=60, le=120) = Field(..., description="Diastolic blood pressure")

    class Config:
        populate_by_name = True

app = FastAPI(
    title="Medical Diagnosis Prediction API",
    description="Predict diagnosis based on patient symptoms and vital signs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model:  Optional[tf.keras.Model] = None
scaler = None
encoder = None
encoded_col_names: Optional[list[str]] = None
feature_columns: Optional[list[str]] = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler, encoder, encoded_col_names, feature_columns

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✓ Model loaded from {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler file not found at {SCALER_PATH}")
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    print(f"✓ Scaler loaded from {SCALER_PATH}")

    if not os.path.exists(ENCODER_PATH):
        raise RuntimeError(f"Encoder file not found at {ENCODER_PATH}")
    try:
        encoder = joblib.load(ENCODER_PATH)
    except Exception:
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
    print(f"✓ Encoder loaded from {ENCODER_PATH}")

    try:
        encoded_col_names = list(encoder.get_feature_names_out(CATEGORICAL_COLS))
    except Exception: 
        encoded_col_names = list(encoder.get_feature_names_out())
    
    feature_columns = NUMERICAL_COLS + encoded_col_names
    print(f"✓ Expected {len(feature_columns)} features: {feature_columns[: 5]}...")

def preprocess_input(payload: PatientInput) -> np.ndarray:

    numerical_values = np.array([[
        payload.Age,
        payload.Heart_Rate_bpm,
        payload.Body_Temperature_C,
        payload.Oxygen_Saturation_percent,
        payload.Systolic,
        payload.Diastolic
    ]])
    
    try:
        numerical_scaled = scaler.transform(numerical_values)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Numerical scaling failed: {str(e)}")
    
    categorical_values = np.array([[
        payload.Gender,  
        payload.Symptom_1,
        payload.Symptom_2,
        payload.Symptom_3
    ]], dtype=object)  
    
    try:
        categorical_encoded = encoder.transform(categorical_values)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Categorical encoding failed: {str(e)}")
    
    if hasattr(categorical_encoded, "toarray"):
        categorical_encoded = categorical_encoded.toarray()
    
    x = np.concatenate([numerical_scaled, categorical_encoded], axis=1)
    
    if x.shape[1] != len(feature_columns):
        raise HTTPException(
            status_code=500,
            detail=f"Feature count mismatch. Expected {len(feature_columns)}, got {x.shape[1]}",
        )
    
    return x

def interpret_prediction(pred_vector: np.ndarray) -> dict:
    class_index = int(np.argmax(pred_vector))
    class_prob = float(np.max(pred_vector))
    
    prob_dict = {
        REVERSE_DIAGNOSIS[i]: float(pred_vector[i])
        for i in range(len(pred_vector))
    }
    
    response = {
        "predicted_diagnosis": REVERSE_DIAGNOSIS[class_index],
        "predicted_class_index":  class_index,
        "confidence": class_prob,
        "all_probabilities": prob_dict
    }
    
    return response

@app.get("/")
def root():
    return {
        "message": "Medical Diagnosis Prediction API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": encoder is not None,
        "feature_count": len(feature_columns) if feature_columns else 0,
        "diagnoses": list(REVERSE_DIAGNOSIS.values())
    }


@app.post("/predict")
def predict(input_data: PatientInput):
    """
    Predict diagnosis based on patient data.
    
    Returns:
        - predicted_diagnosis: The most likely diagnosis
        - predicted_class_index: Index of the predicted class
        - confidence: Probability of the predicted diagnosis
        - all_probabilities:  Probabilities for all diagnoses
    """
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")
    
    try:
        x = preprocess_input(input_data)
    except HTTPException:
        raise
    except Exception as e: 
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
    
    try:
        preds = model.predict(x, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
    
    if preds.ndim == 1:
        pred_vec = preds
    else:
        pred_vec = preds[0]
    
    return interpret_prediction(pred_vec)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)