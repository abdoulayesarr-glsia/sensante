# api/main.py
# SenSante API - Lab 3

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# ==============================
# 1. SCHEMAS PYDANTIC
# ==============================

class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

# ==============================
# 2. CREATION API––
# ==============================

app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical",
    version="0.2.0"
)
from fastapi.middleware.cors import CORSMiddleware

# Autoriser les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En dev : tout accepter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 3. CHARGEMENT DU MODELE
# ==============================

print("Chargement du modèle...")

model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print("Modèle chargé !")
print("Classes :", model.classes_)

# ==============================
# 4. ROUTE HEALTH
# ==============================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "SenSante API is running"
    }

# ==============================
# 5. ROUTE PREDICT
# ==============================

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):

    # Encoder sexe
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}"
        )

    # Encoder region
    try:
        region_enc = le_region.transform([patient.region])[0]
    except:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # Construire les features
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]])

    # Prediction
    diagnostic = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    proba_max = float(proba.max())

    # Niveau de confiance
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # Messages
    messages = {
        "palu": "Suspicion de paludisme. Consultez rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation.",
        "typh": "Suspicion de typhoïde. Consultation nécessaire.",
        "sain": "Pas de pathologie détectée."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un médecin.")
    )