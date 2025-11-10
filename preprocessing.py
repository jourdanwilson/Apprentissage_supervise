{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    " ''' Prétraitement du corpus : Classification des types de questions\n",
    "\n",
    "Ce notebook nettoie le corpus (`corpus_simplifie_131025.xlsx`)\n",
    "et crée le fichier `questions_cleaned_minimal.xlsx`\n",
    "utilisé pour l’apprentissage automatique (totale / partielle / alternative).\n",
    "'''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# 1. Import des bibliothèques\n",
    "# ===============================\n",
    "import pandas as pd\n",
    "import re\n",
    "from pathlib import Path\n",
    "\n",
    "# Définir les chemins\n",
    "data_dir = Path(\"../data\")\n",
    "src_path = data_dir / \"corpus_simplifie_131025.xlsx\"\n",
    "out_path = data_dir / \"questions_cleaned_minimal.xlsx\"\n",
    "\n",
    "print(\"Lecture du fichier :\", src_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# 2. Charger le corpus original\n",
    "# ===============================\n",
    "df = pd.read_excel(src_path)\n",
    "print(\"Colonnes disponibles :\", df.columns.tolist())\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# 3. Nettoyage des questions\n",
    "# ===============================\n",
    "def clean_question(q):\n",
    "    \"\"\"Supprime les marqueurs de locuteur et nettoie les espaces.\"\"\"\n",
    "    if not isinstance(q, str):\n",
    "        return \"\"\n",
    "    q = re.sub(r\"#spk\\d+:\", \"\", q)       # enlever #spk1: / #spk2:\n",
    "    q = re.sub(r\"\\s+\", \" \", q).strip()   # normaliser les espaces\n",
    "    q = re.sub(r\"\\s+([?!,.;:])\", r\"\\1\", q)\n",
    "    return q\n",
    "\n",
    "df[\"question_clean\"] = df[\"question\"].apply(clean_question)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# 4. Sélection et renommage des colonnes\n",
    "# ===============================\n",
    "df_min = df[[\"question_clean\", \"type de question\"]].rename(\n",
    "    columns={\"type de question\": \"type_de_question\"}\n",
    ")\n",
    "\n",
    "# Supprimer les lignes vides ou sans label pertinent\n",
    "df_min = df_min[df_min[\"question_clean\"].str.len() > 0]\n",
    "df_min = df_min[df_min[\"type_de_question\"].isin([\"totale\", \"partielle\", \"alternative\"])]\n",
    "\n",
    "print(\"Nombre d'exemples :\", len(df_min))\n",
    "df_min.sample(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# 5. Sauvegarde du corpus nettoyé\n",
    "# ===============================\n",
    "out_path.parent.mkdir(parents=True, exist_ok=True)\n",
    "df_min.to_excel(out_path, index=False)\n",
    "print(\"Fichier nettoyé sauvegardé :\", out_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "# 6. Analyse rapide (optionnelle)\n",
    "# ===============================\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "df_min[\"type_de_question\"].value_counts().plot(kind=\"bar\", color=\"skyblue\")\n",
    "plt.title(\"Répartition des types de question\")\n",
    "plt.xlabel(\"Type de question\")\n",
    "plt.ylabel(\"Nombre d'occurrences\")\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
