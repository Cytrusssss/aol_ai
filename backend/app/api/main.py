import os
import pickle
import joblib
from typing import Literal, Optional

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


def jaro_distance(s1: str, s2: str) -> float:
    """
    Calculate Jaro distance between two strings.
    Returns a value between 0 (no similarity) and 1 (identical).
    """
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Maximum distance for matches
    match_distance = max(len1, len2) // 2 - 1
    match_distance = max(1, match_distance)
    
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    # Find matches
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]: 
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    # Find transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]: 
            continue
        while not s2_matches[k]: 
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches / len1 + matches / len2 + 
            (matches - transpositions / 2) / matches) / 3
    
    return jaro


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Calculate Jaro-Winkler similarity between two strings. 
    
    Args:
        s1: First string
        s2: Second string
        p: Scaling factor for common prefix (default 0.1, max 0.25)
    
    Returns:
        Similarity score between 0 and 1
    """
    jaro = jaro_distance(s1, s2)
    
    if jaro < 0.7:
        return jaro
    
    # Find common prefix (up to 4 characters)
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i].lower() == s2[i].lower():
            prefix += 1
        else:
            break
    
    return jaro + prefix * p * (1 - jaro)

class PatientInput(BaseModel):
    """Patient input with text-based symptom description"""
    Age: conint(ge=18, le=79) = Field(..., description="Patient age", example=45)
    Gender: Literal["Male", "Female"] = Field(..., example="Male")
    symptoms_description: str = Field(
        ..., 
        description="Describe your symptoms in natural language",
        example="I have a bad cough, high fever, and I feel very tired"
    )
    Heart_Rate_bpm: conint(ge=60, le=120) = Field(..., description="Heart rate in BPM", example=85)
    Body_Temperature_C: confloat(ge=35.5, le=40.0) = Field(..., description="Body temperature in Celsius", example=38.5)
    Oxygen_Saturation_percent: conint(ge=90, le=99) = Field(
        ..., 
        alias="Oxygen_Saturation_%", 
        description="Oxygen saturation percentage",
        example=96
    )
    Systolic: conint(ge=90, le=180) = Field(..., description="Systolic blood pressure", example=120)
    Diastolic: conint(ge=60, le=120) = Field(..., description="Diastolic blood pressure", example=80)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Age": 45,
                "Gender": "Male",
                "symptoms_description": "I have a bad cough, high fever, and I feel very tired",
                "Heart_Rate_bpm": 85,
                "Body_Temperature_C": 38.5,
                "Oxygen_Saturation_%": 96,
                "Systolic": 120,
                "Diastolic": 80
            }
        }

class PatientInputDirect(BaseModel):
    """Patient input with directly selected symptoms (not natural language)"""
    Age: conint(ge=18, le=79) = Field(..., description="Patient age", example=45)
    Gender: Literal["Male", "Female"] = Field(..., example="Male")
    Symptom_1: Literal["Fatigue", "Sore throat", "Body ache", "Shortness of breath", 
                    "Runny nose", "Headache", "Cough", "Fever"] = Field(..., example="Fever")
    Symptom_2: Literal["Fatigue", "Sore throat", "Body ache", "Shortness of breath", 
                    "Runny nose", "Headache", "Cough", "Fever"] = Field(..., example="Cough")
    Symptom_3: Literal["Fatigue", "Sore throat", "Body ache", "Shortness of breath", 
                    "Runny nose", "Headache", "Cough", "Fever"] = Field(..., example="Fatigue")
    Heart_Rate_bpm: conint(ge=60, le=120) = Field(..., description="Heart rate in BPM", example=85)
    Body_Temperature_C:  confloat(ge=35.5, le=40.0) = Field(..., description="Body temperature in Celsius", example=38.5)
    Oxygen_Saturation_percent: conint(ge=90, le=99) = Field(
        ..., 
        alias="Oxygen_Saturation_%", 
        description="Oxygen saturation percentage",
        example=96
    )
    Systolic: conint(ge=90, le=180) = Field(..., description="Systolic blood pressure", example=120)
    Diastolic: conint(ge=60, le=120) = Field(..., description="Diastolic blood pressure", example=80)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Age": 45,
                "Gender": "Male",
                "Symptom_1": "Fever",
                "Symptom_2": "Cough",
                "Symptom_3":  "Fatigue",
                "Heart_Rate_bpm": 85,
                "Body_Temperature_C": 38.5,
                "Oxygen_Saturation_%": 96,
                "Systolic": 120,
                "Diastolic":  80
            }
        }


class PatientInputStructured(BaseModel):
    """Internal model with structured symptoms"""
    Age: int
    Gender: str
    Symptom_1: str
    Symptom_2: str
    Symptom_3: str
    Heart_Rate_bpm: int
    Body_Temperature_C: float
    Oxygen_Saturation_percent: int
    Systolic:  int
    Diastolic:  int

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    """Prediction response with diagnosis and extracted symptoms"""
    predicted_diagnosis:  str = Field(..., description="The predicted diagnosis")
    predicted_class_index: int = Field(..., description="Index of predicted class")
    confidence: float = Field(..., description="Confidence score (0-1)")
    all_probabilities: dict[str, float] = Field(..., description="Probabilities for all diagnoses")
    extracted_symptoms: list[str] = Field(..., description="Symptoms extracted from description")
    extraction_scores: dict[str, float] = Field(..., description="Similarity scores for extracted symptoms")

class PredictionResponseDirect(BaseModel):
    """Prediction response for direct symptom input (no extraction)"""
    predicted_diagnosis:  str = Field(..., description="The predicted diagnosis")
    predicted_class_index: int = Field(..., description="Index of predicted class")
    confidence: float = Field(..., description="Confidence score (0-1)")
    all_probabilities: dict[str, float] = Field(..., description="Probabilities for all diagnoses")

app = FastAPI(
    title="Medical Diagnosis Prediction API",
    description="Predict diagnosis based on natural language symptom description and vital signs",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model:  Optional[tf.keras.Model] = None
scaler = None
encoder = None
encoded_col_names:  Optional[list[str]] = None
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


def extract_symptoms_from_text(text: str, top_n: int = 3, threshold: float = 0.75) -> tuple[list[str], dict[str, float]]:
    """
    Extract top N symptoms from text using Jaro-Winkler similarity. 
    
    Args:
        text: User's symptom description
        top_n: Number of symptoms to extract (default 3)
        threshold: Minimum similarity score (default 0.75)
    
    Returns:
        Tuple of (list of symptoms, dict of scores)
    """
    text_lower = text.lower().strip()
    
    # Clean and tokenize
    for char in [',', '.', '! ', '?', ';', ': ']:
        text_lower = text_lower.replace(char, ' ')
    
    words = [w for w in text_lower.split() if len(w) > 2]
    
    symptom_scores = {}
    
    for symptom in VALID_SYMPTOMS: 
        symptom_lower = symptom.lower()
        
        if symptom_lower in text_lower:
            symptom_scores[symptom] = 1.0
            continue
        
        symptom_words = symptom_lower.split()
        max_score = 0.0
        
        if len(symptom_words) > 1:
            for i in range(len(words) - len(symptom_words) + 1):
                phrase = ' '.join(words[i:i+len(symptom_words)])
                phrase_score = jaro_winkler_similarity(symptom_lower, phrase)
                max_score = max(max_score, phrase_score)
        
        for symptom_word in symptom_words:
            for text_word in words:
                score = jaro_winkler_similarity(symptom_word, text_word)
                max_score = max(max_score, score)
        
        if max_score >= threshold:
            symptom_scores[symptom] = max_score
    
    sorted_symptoms = sorted(symptom_scores.items(), key=lambda x: x[1], reverse=True)
    
    matched_symptoms = [symptom for symptom, score in sorted_symptoms[: top_n]]
    scores_dict = {symptom: score for symptom, score in sorted_symptoms[:top_n]}
    
    default_symptoms = ["Fatigue", "Headache", "Body ache"]
    while len(matched_symptoms) < top_n:
        for default in default_symptoms:
            if default not in matched_symptoms:
                matched_symptoms.append(default)
                scores_dict[default] = 0.0
                break
        if len(matched_symptoms) >= top_n:
            break
    
    return matched_symptoms[: top_n], scores_dict



def preprocess_input(payload: PatientInputStructured) -> np.ndarray:
    """Preprocess input for model prediction"""
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
            detail=f"Feature count mismatch.  Expected {len(feature_columns)}, got {x.shape[1]}",
        )

    return x


def interpret_prediction(pred_vector: np.ndarray) -> dict:
    """Convert model output to readable format"""
    class_index = int(np.argmax(pred_vector))
    class_prob = float(np.max(pred_vector))

    prob_dict = {
        REVERSE_DIAGNOSIS[i]: float(pred_vector[i])
        for i in range(len(pred_vector))
    }

    return {
        "predicted_diagnosis": REVERSE_DIAGNOSIS[class_index],
        "predicted_class_index": class_index,
        "confidence": class_prob,
        "all_probabilities": prob_dict
    }

@app.get("/")
def root():
    import datetime
    print(f"🔥 ROOT ENDPOINT CALLED AT {datetime.datetime.now()}")
    print(f"🔥 TOTAL ROUTES: {len(app.routes)}")
    
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            print(f"   - {route.path} [{route.methods}]")  # This will show ALL routes
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    
    return {
        "message": "Medical Diagnosis Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict_with_sentence": "/predict/sentence",
            "predict_direct_symptoms": "/predict",
            "docs": "/docs"
        },
        "all_routes": routes
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoder_loaded": encoder is not None,
        "feature_count": len(feature_columns) if feature_columns else 0,
        "diagnoses": list(REVERSE_DIAGNOSIS.values()),
        "valid_symptoms": VALID_SYMPTOMS
    }

@app.post("/predict/sentence", response_model=PredictionResponse)
def predict_with_sentence(input_data: PatientInput):
    """
    Predict diagnosis based on natural language symptom description and vital signs.
    
    Simply describe your symptoms naturally (e.g., "I have a cough, fever and headache"),
    and the system will: 
    1. Extract the top 3 matching symptoms using Jaro-Winkler similarity
    2. Analyze your vital signs
    3. Predict the most likely diagnosis
    
    Example input:
    ```json
    {
      "Age": 45,
      "Gender": "Male",
      "symptoms_description": "I have a bad cough, high fever, and I feel very tired",
      "Heart_Rate_bpm":  85,
      "Body_Temperature_C": 38.5,
      "Oxygen_Saturation_%": 96,
      "Systolic": 120,
      "Diastolic": 80
    }
    ```
    """
    print("/predict/sentence endpoint was called!")
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")

    try:
        symptoms, scores = extract_symptoms_from_text(input_data.symptoms_description, top_n=3)
        print(f"📝 Extracted symptoms: {symptoms}")
        print(f"📊 Similarity scores: {scores}")
    except Exception as e: 
        raise HTTPException(status_code=400, detail=f"Symptom extraction failed: {str(e)}")

    structured_input = PatientInputStructured(
        Age=input_data.Age,
        Gender=input_data.Gender,
        Symptom_1=symptoms[0],
        Symptom_2=symptoms[1],
        Symptom_3=symptoms[2],
        Heart_Rate_bpm=input_data.Heart_Rate_bpm,
        Body_Temperature_C=input_data.Body_Temperature_C,
        Oxygen_Saturation_percent=input_data.Oxygen_Saturation_percent,
        Systolic=input_data.Systolic,
        Diastolic=input_data.Diastolic
    )

    try:
        x = preprocess_input(structured_input)
    except HTTPException: 
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

    try: 
        preds = model.predict(x, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    pred_vec = preds[0] if preds.ndim > 1 else preds

    result = interpret_prediction(pred_vec)
    result["extracted_symptoms"] = symptoms
    result["extraction_scores"] = scores

    return result

@app.post("/predict", response_model=PredictionResponseDirect)  
def predict_diagnosis(input_data: PatientInputDirect):
    """
    Predict diagnosis with directly selected symptoms (not natural language).
    
    Use this endpoint when you already know the exact symptoms to select.
    For natural language input, use /predict/sentence instead.
    
    Example input:
    ```json
    {
      "Age": 45,
      "Gender": "Male",
      "Symptom_1": "Fever",
      "Symptom_2": "Cough",
      "Symptom_3":  "Fatigue",
      "Heart_Rate_bpm": 85,
      "Body_Temperature_C": 38.5,
      "Oxygen_Saturation_%": 96,
      "Systolic": 120,
      "Diastolic":  80
    }
    ```
    
    Returns:
        - predicted_diagnosis: The most likely diagnosis
        - confidence: Probability of the predicted diagnosis
        - all_probabilities:  Probabilities for all diagnoses
    """
    if model is None or scaler is None or encoder is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")
    
    # Convert PatientInputDirect to PatientInputStructured
    structured_input = PatientInputStructured(
        Age=input_data.Age,
        Gender=input_data.Gender,
        Symptom_1=input_data.Symptom_1,
        Symptom_2=input_data.Symptom_2,
        Symptom_3=input_data.Symptom_3,
        Heart_Rate_bpm=input_data.Heart_Rate_bpm,
        Body_Temperature_C=input_data.Body_Temperature_C,
        Oxygen_Saturation_percent=input_data.Oxygen_Saturation_percent,
        Systolic=input_data.Systolic,
        Diastolic=input_data.Diastolic
    )
    
    # Preprocess
    try:
        x = preprocess_input(structured_input)
    except HTTPException:
        raise
    except Exception as e: 
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
    
    # Predict
    try:
        preds = model.predict(x, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
    
    pred_vec = preds[0] if preds.ndim > 1 else preds
    
    return interpret_prediction(pred_vec)


if __name__ == "__main__": 
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)