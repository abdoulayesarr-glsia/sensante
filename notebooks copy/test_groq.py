# notebooks/test_groq.py

# Test de l'API Groq avec Llama 3

import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables du fichier .env
load_dotenv()

# Récupérer la clé API
api_key = os.getenv("GROQ_API_KEY")

# Vérifier si la clé existe
if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvée dans le fichier .env")
    exit()

# Créer le client Groq
client = Groq(api_key=api_key)

try:
    # Envoyer une requête au modèle
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant médical sénégalais. "
                    "Réponds en français simple. "
                    "Maximum 3 phrases."
                ),
            },
            {
                "role": "user",
                "content": "Quels sont les symptômes du paludisme ?",
            },
        ],
        max_tokens=200,
        temperature=0.3,
    )

    # Afficher la réponse
    print("=== Réponse de Llama 3 ===")
    print(response.choices[0].message.content)

    # Afficher les tokens utilisés
    print(f"\nTokens utilisés : {response.usage.total_tokens}")

except Exception as e:
    print("Erreur lors de l'appel API :")
    print(e)