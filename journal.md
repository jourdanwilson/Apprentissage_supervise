# Journal – Branche JWilson

**Date :** 10 novembre 2025  

## Étapes réalisées
- Création du dossier de projet local et du dépôt GitHub
- Connexion du dépôt local au dépôt distant (`git remote add origin ...`)
- Création de la branche `JWilson_branche`
- Ajout du fichier `README.md`
- Ajout du notebook `preprocessing.ipynb` pour le nettoyage du corpus
- Génération du fichier `questions_cleaned_minimal.xlsx` à partir du corpus original

## Idées et hypothèses
- Tester deux versions du modèle :
  1. Basée uniquement sur la question (`question_clean`)
  2. Basée sur la question + contexte (`previous_context`)
- Hypothèse : le contexte pourrait légèrement améliorer la détection des questions partielles,
  mais risque d’ajouter du bruit.
- À discuter : faut-il intégrer le champ `speaker` dans le modèle ?
- Idée : ajouter un graphique de répartition dans le notebook preprocessing.

  ## Prochaines étapes
- [ ] Finaliser le nettoyage et vérifier le fichier `questions_cleaned_minimal.xlsx`
- [ ] Implémenter le modèle TF-IDF + Logistic Regression
- [ ] Évaluer les performances par type de question
- [ ] Comparer les résultats avec et sans contexte
- [ ] Documenter les résultats pour la présentation
