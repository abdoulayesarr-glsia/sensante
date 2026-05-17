# api/main.py
# SenSante API - Lab 3

import os
import joblib
import numpy as np

from dotenv import load_dotenv
from groq import Groq

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

# =====================================
# 1. CHARGER LES VARIABLES D'ENVIRONNEMENT
# =====================================

load_dotenv()

# =====================================
# 2. CLIENT GROQ
# =====================================

groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialisé.")
else:
    print(
        "ATTENTION : GROQ_API_KEY non trouvée. "
        "/explain sera désactivé."
    )

# =====================================
# 3. CREATION API
# =====================================

app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical",
    version="0.2.0"
)

# =====================================
# 4. CORS
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# 5. PROMPT SYSTEME
# =====================================

SYSTEM_PROMPT = """
Tu es un assistant médical sénégalais.

Tu reçois un diagnostic et des données patient.
Explique le résultat en français simple,
comme un médecin parlerait à son patient.

Sois rassurant mais recommande toujours
une consultation médicale.

Maximum 3 phrases.

Ne fais JAMAIS de diagnostic toi-même.
Tu expliques uniquement le diagnostic fourni.
"""

# =====================================
# 6. SCHEMAS PYDANTIC
# =====================================

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


class ExplainInput(BaseModel):
    diagnostic: str = Field(
        ...,
        description="Diagnostic prédit par le modèle"
    )

    probabilite: float = Field(
        ...,
        description="Probabilité du diagnostic"
    )

    age: int = Field(...)

    sexe: str = Field(...)

    temperature: float = Field(...)

    region: str = Field(...)


class ExplainOutput(BaseModel):
    explication: str = Field(
        ...,
        description="Explication en français"
    )

    modele_llm: str = Field(
        default="llama-3.1-8b-instant",
        description="Modèle LLM utilisé"
    )

# =====================================
# 7. CHARGEMENT DU MODELE
# =====================================

print("Chargement du modèle...")

model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print("Modèle chargé !")
print("Classes :", model.classes_)

# =====================================
# 8. ROUTE HEALTH
# =====================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "SenSante API is running"
    }

# =====================================
# 9. ROUTE PREDICT
# =====================================

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

    # Encoder région
    try:
        region_enc = le_region.transform([patient.region])[0]
    except:
        return DiagnosticOutput(
            diagnostic="erreur",
            probabilite=0.0,
            confiance="aucune",
            message=f"Région inconnue : {patient.region}"
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

    # Prédiction
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
        message=messages.get(
            diagnostic,
            "Consultez un médecin."
        )
    )

# =====================================
# 10. ROUTE EXPLAIN
# =====================================

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):

    if not groq_client:
        return ExplainOutput(
            explication=(
                "Service d'explication indisponible. "
                "Clé API non configurée."
            ),
            modele_llm="aucun"
        )

    # Construire le prompt utilisateur
    user_prompt = (
        f"Patient : {data.sexe}, "
        f"{data.age} ans, région {data.region}\n"
        f"Température : {data.temperature} °C\n"
        f"Diagnostic du modèle : {data.diagnostic} "
        f"(probabilité {data.probabilite:.0%})\n"
        f"Explique ce résultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=200,
            temperature=0.3
        )

        explication = response.choices[0].message.content

        return ExplainOutput(
            explication=explication,
            modele_llm="llama-3.1-8b-instant"
        )

    except Exception as e:
        return ExplainOutput(
            explication=f"Erreur lors de l'appel au LLM : {str(e)}",
            modele_llm="erreur"
        )