# LAB 2 SEN SANTE
# Entrainement + Serialization
# =========================

import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# 1 Charger dataset
# -------------------------

df = pd.read_csv("data/patients_dakar.csv")

print("Dimensions :", df.shape)
print(df.head())


# -------------------------
# 2 Encoder les variables
# -------------------------

le_sexe = LabelEncoder()
le_region = LabelEncoder()

df["sexe_encoded"] = le_sexe.fit_transform(df["sexe"])
df["region_encoded"] = le_region.fit_transform(df["region"])


feature_cols = [
    "age",
    "sexe_encoded",
    "temperature",
    "tension_sys",
    "toux",
    "fatigue",
    "maux_tete",
    "region_encoded"
]

X = df[feature_cols]
y = df["diagnostic"]


# -------------------------
# 3 Train / Test
# -------------------------

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# 4 Entrainer modele
# -------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

print("Modele entrainé")


# -------------------------
# 5 Evaluation
# -------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("Accuracy :",accuracy)


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print(cm)

print(classification_report(
    y_test,
    y_pred
))


# -------------------------
# Matrice confusion
# -------------------------

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Reel")

os.makedirs("figures",exist_ok=True)

plt.savefig("figures/confusion_matrix.png")
plt.show()



# -------------------------
# 6 Serialisation
# -------------------------

os.makedirs("models",exist_ok=True)

joblib.dump(model,"models/model.pkl")

joblib.dump(
    le_sexe,
    "models/encoder_sexe.pkl"
)

joblib.dump(
    le_region,
    "models/encoder_region.pkl"
)

joblib.dump(
    feature_cols,
    "models/feature_cols.pkl"
)

print("Modele serialisé")


# -------------------------
# 7 Test modele recharge
# -------------------------

model_loaded=joblib.load(
"models/model.pkl"
)

nouveau_patient={

'age':28,
'sexe':'F',
'temperature':39.5,
'tension_sys':110,
'toux':True,
'fatigue':True,
'maux_tete':True,
'region':'Dakar'

}

sexe_enc=le_sexe.transform(
[nouveau_patient['sexe']]
)[0]

region_enc=le_region.transform(
[nouveau_patient['region']]
)[0]

features=[

nouveau_patient['age'],
sexe_enc,
nouveau_patient['temperature'],
nouveau_patient['tension_sys'],
int(nouveau_patient['toux']),
int(nouveau_patient['fatigue']),
int(nouveau_patient['maux_tete']),
region_enc

]

diagnostic=model_loaded.predict(
[features]
)[0]

print("Diagnostic :",diagnostic)


# -------- EXERCICE 1 --------

print("\nImportance des features:")

importances=model.feature_importances_

for name,imp in sorted(
zip(feature_cols,importances),
key=lambda x:x[1],
reverse=True
):
    print(f"{name:20s} : {imp:.3f}")


# -------- EXERCICE 2 --------

patients=[

[20,1,37.0,120,0,0,0,0],
[35,0,40.2,110,1,1,1,0],
[70,1,38.5,130,1,1,0,0]

]

for i,p in enumerate(patients,1):
    pred=model.predict([p])[0]
    print(f"Patient {i} --> {pred}"
    
)