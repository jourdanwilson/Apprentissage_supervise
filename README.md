# Projet Apprentissage Supervisé
Classification des types de questions (Totale / Partielle / Alternative)

Projet de groupe pour travailler sur un corpus annoté de questions orales.
Objectif : entraîner et évaluer un modèle qui prédit le type de question (totale, partielle, alternative)
à partir de l'énoncé, et tester l'apport du contexte.


## Description

Ce projet a pour but de classifier automatiquement les types de questions en français dans un corpus oral :

- **Totale** → réponse attendue oui/non  
- **Partielle** → réponse attendue contenant une information spécifique (où, qui, pourquoi…)  
- **Alternative** → réponse attendue parmi un choix (thé **ou** café ?)

## **Distribution des étiquettes:**

![Distribution des étiquettes](https://cdn.discordapp.com/attachments/1419624042981363812/1437437597537533952/image.png?ex=69133d95&is=6911ec15&hm=bb40ed4600a8d9caa9e6ba5d8096314c58e033f923d1701b8e2e4f3a0463f5b5)<img width="665" height="602" alt="image" src="https://github.com/user-attachments/assets/aa254b04-167a-4fe5-b912-6fcb0b167a1f" />


Nous utilisons un corpus annoté (`corpus_simplifie_131025.xlsx`) fourni par Yeojun Yun (MoDyCo).

## Data
├── corpus_simplifie_131025.xlsx # Corpus original <br>
├── questions_cleaned_minimal.xlsx # Corpus nettoyé (question + type) <br>

src/ <br>
├── preprocess.py # Nettoyage + extraction des questions <br>
├── train_model.py # Entraînement + sauvegarde du modèle <br>

## Méthodologie

1. **Nettoyage**
   - Suppression des marqueurs `#spk1:` et `#spk2:`
   - Extraction des questions et des étiquettes (`type de question`)

2. **Vectorisation**
   - `TfidfVectorizer(stop_words='french', ngram_range=(1,2))`

3. **Modélisation**
   - Baseline : Logistic Regression
   - Option avancée : CamemBERT (transformers)

4. **Évaluation**
   - Accuracy / F1-score par catégorie (important car classes déséquilibrées)

---


## Équipe

- …
-
- annotations : d'après Yeojun Yun (Université Paris Nanterre, MoDyCo)

