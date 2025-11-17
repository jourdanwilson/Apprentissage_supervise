```
Extract_info_excel.py
```
Script permettant de récupérer les informations contenus dans le fichier excel source.
Il crée un fichier JSON contenant les informations suivantes :

* id (identifiant)
* file_name (le nom du fichier d'ou provient les annotations)
* input_text (la fusion entre le contexte précédent la question et la question en elle-même.)
* label (le type de question)
* intention (l'intention derrière la question posée)

-----


```
Stats&traitement.py
```

Script permettant de séparer le contexte et la question de l'input_text, procéder à une analyse du corpus JSON permettant de déterminer le nombre de questions, de labels, d'intentions uniques et totales.
Il procède également au nettoyage du corpus et à sa lemmatisation afin de déterminer les lemmes présents dans les questions, leurs nombres, leurs fréquences d'apparitions, le nombre de mot en moyenne par question, la représentation du "Part of Speech" et un classement des 5 lemmes les plus représentés.

Il permet également après analyse du corpus, de fournir des graphiques ces mêmes analyses et si selon la représentation des labels, un déséquilibre est détecté, permet de charger un second corpus (.JSON) qui sera fusionné au premier afin d'augmenter le corpus.

Enfin, il donne les statistiques finales du corpus (après fusion).

-----

```
alt_question_generator.py
```
Script permettant, à partir de l'API d'Open AI (et une clé), de générer un certain nombre de questions de type alternatives (déterminé par l'utilisateur) afin de constituer un second corpus au format JSON (que l'on fusionnera par la suite avec le premier corpus afin d'obtenir notre corpus dans sa forme finale).

-----

PS : Les trois scripts mentionnés ci-dessus sont dotés d'une interface Qt afin de faciliter leur utilisation auprès du plus grand nombre.

-----

## A venir:

- Un script séparant le corpus en deux entités, une dédié à l'entrainement du modèle, le second pour le test du modèle. Il apportera également les différentes analyses nécessaires à l'analyse de la performance du modèle (graphiques et résumé textuels).

- Un fichier "requirements.txt" (pour les modules nécessaires à l'exécution des scripts)



Note : Les graphiques et documents récapitulatifs des statistiques du corpus datant du 10 Novemebre ne sont pas encore définitif (contenant que le corpus sans l'ajout supplémentaire des questions alternatives)

