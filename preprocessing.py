 ''' Prétraitement du corpus : Classification des types de questions

Ce notebook nettoie le corpus (`corpus_simplifie_131025.xlsx`)
et crée le fichier `questions_cleaned_minimal.xlsx`
utilisé pour l’apprentissage automatique (totale / partielle / alternative).
'''
     

# 1. Import des bibliothèques
import pandas as pd
import re
from pathlib import Path

# Définir les chemins
data_dir = Path("../data")
src_path = data_dir / "corpus_simplifie_131025.xlsx"
out_path = data_dir / "questions_cleaned_minimal.xlsx"

print("Lecture du fichier :", src_path)
     

# 2. Charger le corpus original
df = pd.read_excel(src_path)
print("Colonnes disponibles :", df.columns.tolist())
df.head()
     

# 3. Nettoyage des questions
def clean_question(q):
    """Supprime les marqueurs de locuteur et nettoie les espaces."""
    if not isinstance(q, str):
        return ""
    q = re.sub(r"#spk\d+:", "", q)       # enlever #spk1: / #spk2:
    q = re.sub(r"\s+", " ", q).strip()   # normaliser les espaces
    q = re.sub(r"\s+([?!,.;:])", r"\1", q)
    return q

df["question_clean"] = df["question"].apply(clean_question)
     

# 4. Sélection et renommage des colonnes
df_min = df[["question_clean", "type de question"]].rename(
    columns={"type de question": "type_de_question"}
)

# Supprimer les lignes vides ou sans label pertinent
df_min = df_min[df_min["question_clean"].str.len() > 0]
df_min = df_min[df_min["type_de_question"].isin(["totale", "partielle", "alternative"])]

print("Nombre d'exemples :", len(df_min))
df_min.sample(5)
     

# 5. Sauvegarde du corpus nettoyé
out_path.parent.mkdir(parents=True, exist_ok=True)
df_min.to_excel(out_path, index=False)
print("Fichier nettoyé sauvegardé :", out_path)
     

# 6. Analyse rapide (optionnelle)
import matplotlib.pyplot as plt

df_min["type_de_question"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Répartition des types de question")
plt.xlabel("Type de question")
plt.ylabel("Nombre d'occurrences")
plt.show()
