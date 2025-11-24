## Documentation:

*Partie Corpus :*

> original_corpus.json

Corpus crée par l'extraction de ses données à partir d'un fichier Excel.
Il contient 218 questions totales, 81 questions partielles et 20 questions alternatives.

> alternatives_questions.json

Corpus intégralement composé de questions alternatives (61) afin de rééquilibrer le dataset.

> final_corpus.json

Fusion des deux corpus ci-dessus créant notre corpus final, utilisé pour l'entrainement.

----

*Partie Résultats :*

> 81-81-81

Dossier entièrement dédié aux résultats de l'entrainement pour nos classes divisés en 81/81/81 (81 questions totales, 81 questions partielles et 81 questions alternatives).

Dossier contenant les graphiques (matrices de confusions,histogrammes), rapports de classifications de chaque classifieur utilisé, un table csv contenant les résultats de la précision et de la macro F1 pour chaque modèle et le modèle le plus performant.

> 150-81-81

Dossier entièrement dédié aux résultats de l'entrainement pour nos classes divisés en 150/81/81 (150 questions totales, 81 questions partielles et 81 questions alternatives).

Dossier contenant les graphiques (matrices de confusions,histogrammes), rapports de classifications de chaque classifieur utilisé, un table csv contenant les résultats de la précision et de la macro F1 pour chaque modèle et le modèle le plus performant.

> stats

Dossier contenant les graphiques et le résumé textuel des statistiques de fréquences d'intentions, fréquences de labels et le décompte du POS. Cela, en deux versions, l'une avant la fusion des deux corpus et l'autre après.