import sys
import json
import os
import re
import spacy
import matplotlib.pyplot as plt
from collections import Counter
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PySide6.QtGui import QFont

# Modèle SpaCy pour le français
try:
    nlp = spacy.load("fr_core_news_md")
except:
    raise RuntimeError("Le modèle SpaCy 'fr_core_news_md' n'est pas installé. Veuillez l'installer en exécutant 'python -m spacy download fr_core_news_md' dans votre terminal.")

# Fonction pour séparer le contexte et la question
def split_context_and_question(input_text: str):
    
    parts = re.split(r"Question:", input_text, maxsplit=1)
    if len(parts) == 2:
        context = re.sub(r"(Locuteur \d+:)", "", parts[0])
        question = re.sub(r"(Locuteur \d+:)", "", parts[1])
    else:
        context = ""
        question = re.sub(r"(Locuteur \d+:)", "", input_text)


    context = re.sub(r"\s+", " ", context).strip()
    question = re.sub(r"\s+", " ", question).strip()
    
    return context, question

# Analyse du corpus JSON   
def analyze_corpus(data):
    total_questions = len(data)
    label_counts = Counter(entry["label"] for entry in data if "label" in entry)
    intention_counts = Counter(entry["intention"] for entry in data if "intention" in entry)
    unique_intentions = len(intention_counts)
    return total_questions, label_counts, intention_counts, unique_intentions

def freq_labels_ok(label_counts):
    if not label_counts:
        return True
    majority_count = max(label_counts.values())
    for label, count in label_counts.items():
        if count < majority_count / 2:
            return False
    return True

# Fusion de deux corpus JSON
def fusion_corpus(data1, data2):
    return data1 + data2

# Nettoyage et lemmatisation des questions
def clean_and_lemmatize(data):
    lemmas_count = []
    word_per_question = []
    pos_counter = Counter()

    for entry in data:
        question = entry.get("question", "")

        doc = nlp(question)

        lemmas_question = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
        tokens_question = [token.text for token in doc if not token.is_punct and not token.is_space]
        lemmas_count.extend(lemmas_question)
        word_per_question.append(len(tokens_question))

        for token in doc:
            if not token.is_punct and not token.is_space:
                pos_counter[token.pos_] += 1

        entry["lemmas"] = lemmas_question
        entry["nb_tokens"] = len(tokens_question)

    top_5_lemmas = Counter(lemmas_count).most_common(5)
    avg_words_per_question = sum(word_per_question) / len(word_per_question) if word_per_question else 0

    return {
        "top_5_lemmas": top_5_lemmas,
        "pos_distribution": dict(pos_counter),
        "avg_words_per_question": avg_words_per_question
    }

# Création de graphiques
def plot_bar_chart(counter, title, xlabel, ylabel, filepath):
    items = list(counter.items())
    if not items:
        return
    items.sort(key=lambda x: x[1], reverse=True)
    labels, values = zip(*items)

    labels = [str(label) for label in labels]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

class StatsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyse de Corpus JSON")
        self.setGeometry(300, 300, 600, 400)
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label_title = QLabel("Statistiques de corpus JSON")
        self.label_title.setFont(QFont("Arial", 16, QFont.Bold))
        self.layout.addWidget(self.label_title)

        self.stats_label = QLabel("Aucune donnée chargée.")
        self.layout.addWidget(self.stats_label)
        
        self.load_button = QPushButton("Charger un fichier JSON")
        self.load_button.clicked.connect(self.load_first_json)
        self.layout.addWidget(self.load_button)

        self.data1 = None
        self.data2 = None
        self.final_data = None

        self.reset_button = QPushButton("Réinitialiser")
        self.reset_button.clicked.connect(self.reset_app)
        self.layout.addWidget(self.reset_button)

    # Affichage des statistiques (après premier chargement)
    def show_stats(self, total_questions, label_counts, intention_counts, unique_intentions):
        stats_text = (
            f"Total de questions: {total_questions}\n"
            "Répartition des labels:\n"
        )
        for label, count in label_counts.items():
            stats_text += f"  - {label}: {count}\n"

        stats_text += "\nIntentions :\n"
        for intention, count in intention_counts.items():
            stats_text += f"  - {intention}: {count}\n"

        stats_text += f"\nNombre d'intentions uniques: {unique_intentions}\n"
        self.stats_label.setText(stats_text)
        

    def load_first_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Ouvrir le premier fichier JSON", os.getcwd(), "Fichiers JSON (*.json)")
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier JSON:\n{e}")
            return
        
        self.data1 = data
        QMessageBox.information(self, "Succès", "Premier fichier JSON chargé avec succès.")

        # Vérification de la nécessité de transformation
        needs_transformation = any("input_text" in entry for entry in self.data1)

        if needs_transformation:
            for entry in self.data1:
                if "input_text" in entry:
                    context, question = split_context_and_question(entry["input_text"])
                    entry["context"] = context
                    entry["question"] = question
                    del entry["input_text"]

            # Sauvegarde du corpus contexte/question séparé si nécessaire.
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Enregistrer le corpus transformé (avec contexte/question)",
                "",
                "Fichier JSON (*.json)"
            )
            if save_path:
                if not save_path.lower().endswith(".json"):
                    save_path += ".json"

                try:
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(self.data1, f, ensure_ascii=False, indent=2)
                    QMessageBox.information(
                        self,
                        "Succès",
                        f"Corpus transformé sauvegardé dans :\n{save_path}"
                    )
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Erreur",
                        f"Impossible d'enregistrer le corpus transformé :\n{e}"
                    )
        else:
            QMessageBox.information(self, "Information", "Le corpus ne nécessite pas de transformation.")
        
        total_questions, label_counts, intention_counts, unique_intentions = analyze_corpus(data)
        self.show_stats(total_questions, label_counts, intention_counts, unique_intentions)

        

        if not freq_labels_ok(label_counts):
            reponse = QMessageBox.question(self, "Corpus incomplet", "Une ou plusieurs classes ont une fréquence inférieure à la moitié de la classe majoritaire.\n" "Voulez-vous charger un second corpus pour compléter ?", QMessageBox.Yes | QMessageBox.No)
            if reponse == QMessageBox.Yes:
                self.load_second_json()
            else:
                self.final_data = self.data1
                self.analyze_and_last_show()

        else:
            self.final_data = self.data1
            self.analyze_and_last_show()

    def load_second_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choisir le second fichier JSON", "", "Fichiers JSON (*.json)")
        if not file_path:
            QMessageBox.information(self, "Information", "Pas de second corpus chargé.")
            self.final_data = self.data1
            self.analyze_and_last_show()
            return
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data2 = json.load(f)
                if isinstance(data2, dict):
                    data2 = [data2]
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier JSON:\n{e}")
            self.final_data = self.data1
            self.analyze_and_last_show()
            return
        
        self.data2 = data2

        total_q2, label_counts2, intention_counts2, unique_int2 = analyze_corpus(data2)
        stats_text = (
            f"Second corpus chargé:\nNombre total de questions : {total_q2}\n\n"
            f"Répartitions des labels :\n"
        )

        for label, count in label_counts2.items():
            stats_text += f"  - {label}: {count}\n"

        stats_text += "\nIntentions :\n"
        for intention, count in intention_counts2.items():
            stats_text += f"  - {intention}: {count}\n"

        stats_text += f"\nNombre d'intentions uniques: {unique_int2}\n"
        
        QMessageBox.information(self, "Statistiques du second corpus", stats_text)

        reponse = QMessageBox.question(self, "Fusion de corpus", "Voulez-vous fusionner le premier et second corpus et l'enregistrer?", QMessageBox.Yes | QMessageBox.No)
        if reponse == QMessageBox.Yes:
            self.final_data = fusion_corpus(self.data1, self.data2)

            save_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le corpus fusionné", "", "Fichier JSON (*.json)")
            if save_path:
                if not save_path.lower().endswith(".json"):
                    save_path += ".json"

                try:
                    for entry in self.final_data:
                        if "context" in entry and "question" in entry:
                            continue

                        if "input_text" in entry:
                            context, question = split_context_and_question(entry["input_text"])
                            entry["context"] = context
                            entry["question"] = question
                            del entry["input_text"]

                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(self.final_data, f, ensure_ascii=False, indent=2)
                        QMessageBox.information(self, "Succès", f"Corpus fusionné enregistré dans : \n{save_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Erreur",f"Erreur de sauvegarde :\n{e}")
            self.analyze_and_last_show()
        else:
            self.final_data = self.data1
            self.analyze_and_last_show()

    # Affichage des statistiques finales (après fusion)
    def analyze_and_last_show(self):
        if not self.final_data:
            QMessageBox.warning(self, "Aucun corpus", "Aucun corpus chargé pour l'analyse finale.")
            return
        
        stats_ling = clean_and_lemmatize(self.final_data)
        total_questions, label_counts, intention_counts, unique_intentions = analyze_corpus(self.final_data)

        txt = (
            f"Nombre total de questions (après fusion) : {total_questions}\n\n"
            f"Répartitions des labels :\n"
        )

        for label, count in label_counts.items():
            txt += f" - {label}: {count}\n"

        txt += "\nIntentions :\n"
        for intention, count in intention_counts.items():
            txt += f"  - {intention}: {count}\n"

        txt += f"\nNombre d'intentions différentes : {unique_intentions}\n"

        txt += "\nStatistiques linguistiques :\n"
        txt += f"Nombre moyen de mots par question : {stats_ling['avg_words_per_question']:.2f}\n"
        txt += "Top 5 des lemmes les plus fréquents :\n"
        for lemma, freq in stats_ling["top_5_lemmas"]:
            txt += f"  - {lemma}: {freq}\n"

        txt += "\nRépartition POS (part-of-speech) :\n"
        for pos, count in stats_ling["pos_distribution"].items():
            txt += f"  - {pos}: {count}\n"

        self.stats_label.setText(txt)

        reponse = QMessageBox.question(self, "Sauvegarde", "Voulez-vous sauvegarder les graphiques et le résumé textuel ?", QMessageBox.Yes | QMessageBox.No)
        if reponse == QMessageBox.Yes:
            self.save_graphs_resume(label_counts, intention_counts, stats_ling)

    def save_graphs_resume(self, label_counts, intention_counts, stats_ling):
        dossier = QFileDialog.getExistingDirectory(self, "Choisir le dossier pour la sauvegarde")
        if not dossier:
            QMessageBox.information(self, "Annulé", "Sauvegarde annulée.")
            return
        
        try:
            plot_bar_chart(label_counts, "Fréquence des types de questions", "Types de questions", "Nombre", os.path.join(dossier, "freq_labels.png"))
            plot_bar_chart(intention_counts, "Fréquence des intentions", "Intentions", "Nombre", os.path.join(dossier, "freq_intentions.png"))

            plot_bar_chart(stats_ling["pos_distribution"], "Répartition POS", "POS", "Nombre", os.path.join(dossier, "pos_counts.png"))
        except Exception as e:
            QMessageBox.warning(self, "Erreur graphique", f"Erreur lors de la création des graphiques:\n{e}")

        try:
            with open(os.path.join(dossier, "resume_stats.txt"), "w", encoding="utf-8") as f:
                f.write(self.stats_label.text())
            QMessageBox.information(self, "Succès", "Graphiques et résumé sauvegardés.")
        except Exception as e:
            QMessageBox.warning(self, "Erreur sauvegarde", f"Erreur lors de la sauvegarde du résumé:\n{e}")

    def reset_app(self):
        self.data1 = None
        self.data2 = None
        self.final_data = None
        self.stats_label.setText("Aucune donnée chargée.")
        QMessageBox.information(self, "Réinitialisation", "L'application a été réinitialisée.")

def main():
        app = QApplication(sys.argv)
        window = StatsApp()
        window.show()
        sys.exit(app.exec())

if __name__ == "__main__":
    main()

        