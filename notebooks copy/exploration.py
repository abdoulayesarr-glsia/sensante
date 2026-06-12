"""
SenSante - Exploration du dataset patients_dakar.csv
Lab 1 : Git, Python et Structure Projet
"""

import pandas as pd

# ===== CHARGER LES DONNEES =====
df = pd.read_csv("data/patients_dakar.csv")

# ===== PREMIERS APERCUS =====
print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

# Dimensions du dataset
print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

# Aperçu des 5 premières lignes
print("\n--- 5 premiers patients ---")
print(df.head())

# ===== STATISTIQUES DE BASE =====
print("\n--- Statistiques descriptives ---")
print(df.describe().round(2))

# ===== REPARTITION DES DIAGNOSTICS =====
print("\n--- Repartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts()

for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f"{diag:12s} : {count:3d} patients ({pct:.1f}%)")

# ===== REPARTITION PAR REGION =====
print("\n--- Repartition par region (top 5) ---")
region_counts = df["region"].value_counts().head(5)

for region, count in region_counts.items():
    print(f"{region:15s} : {count:3d} patients")

# ===== TEMPERATURE MOYENNE PAR DIAGNOSTIC =====
print("\n--- Temperature moyenne par diagnostic ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()

for diag, temp in temp_by_diag.items():
    print(f"{diag:12s} : {temp:.1f} C")
    
   # ===== PATIENTS PAR SEXE ET PAR DIAGNOSTIC =====
print("\n--- Nombre de patients par sexe et par diagnostic ---")

patients_par_sexe_diag = df.groupby(["sexe", "diagnostic"]).size()

for (sexe, diag), count in patients_par_sexe_diag.items():
    print(f"{sexe:6s} | {diag:12s} : {count} patients")

print("\n" + "=" * 50)
print("Exploration terminee !")
print("Prochain lab : entrainer un modele ML")
print("=" * 50)