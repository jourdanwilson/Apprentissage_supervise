```
extract_excel.py
```
Script permettant de récupérer les informations contenus dans le fichier Excel source de nos données.
Il propose une prévisualisation du fichier JSON avant son exportation.
Il crée un fichier JSON contenant les informations suivantes :

* id (identifiant)
* file_name (le nom du fichier d'ou provient les annotations)
* input_text (la fusion entre le contexte précédent la question et la question en elle-même.)
* label (le type de question)
* intention (l'intention derrière la question posée)

-----


```
pretreatment.py
```

Script permettant de séparer le contexte et la question de l'input_text, procéder à une analyse du corpus JSON permettant de déterminer le nombre de questions, de labels, d'intentions uniques et totales.
Il procède également au nettoyage du corpus et à sa lemmatisation afin de déterminer les lemmes présents dans les questions, leurs nombres, leurs fréquences d'apparitions, le nombre de mot en moyenne par question, la représentation du "Part of Speech" et un classement des 5 lemmes les plus représentés.

Il permet également après analyse du corpus, de fournir des graphiques (visibles sur l'interface et sauvegardé à la fin) de ces mêmes analyses et si, selon la représentation des labels, un déséquilibre est détecté, le script permet de charger un second corpus (.JSON) qui sera fusionné au premier afin d'augmenter les données.

Le script nous donne les statistiques cités précedmment du premier corpus seul et celui fusionné si ce dernier a été crée durant l'execution du programme.

-----

```
alt_questions_generator.py
```
Script permettant, à partir de l'API d'Open AI (et une clé), de générer un certain nombre de questions de type alternatives (déterminé par l'utilisateur) afin de constituer un second corpus au format JSON (que l'on fusionnera par la suite avec le premier corpus afin d'obtenir notre corpus dans sa forme finale).

-----

```
model_trainer.py
```
Script permettant de charger un corpus structuré (ici, en .JSON), détecte le nombre d'entrées, laisser le choix à l'utilisateur entre un entrainement de modèles uniquement sur les questions ou questions + contextes et génère des graphiques, résumés textuels et en déduit le meilleur modèle parmi ceux séléctionnés.

L'utilisateur a la possibilité de sous-échantillonner (undersampler) certaines classes afin d'avoir un dataset plus équilibré.

Après, se trouve le choix des classifieurs. Nous avons ici :

* LinearSVC
* Logistic Regression
* Naive Bayes
* Random Forest

L'utilisateur choisit le dossier de sauvegarde des graphiques et résumés textuels générés par le script. Il peut également choisir si le script exporte simplement le meilleur modèle parmi ceux séléctionnés ou tous les graphiques et résumés de chaque classifieur.

-----

```
model_trainer_camembert.py
```
Même principe que le script précédent mais cette fois-ci dédié intégralement à l'apprentissage profond et au modèle CamemBERT. Ce script a globalement les mêmes caractéristiques que le précédent (sous-échantillonage personnalisé, choix du corpus, choix d'ajout ou d'exclusion du contexte, choix du dossier de sauvegarde, entrainement, affichage des résultats et sauvegarde des graphiques et le résumé textuel). Ce script ne propose pas de sauvegarder le modèle, pour soucis d'espace.

-----

PS : Les cinq scripts mentionnés ci-dessus sont dotés d'une interface Qt afin de faciliter leur utilisation auprès du plus grand nombre.







