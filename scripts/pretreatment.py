import sys
import json
import os
import re
import spacy
import matplotlib.pyplot as plt
from collections import Counter
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox,
    QTabWidget, QGroupBox, QStatusBar, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Modèle SpaCy pour le français
try:
    nlp = spacy.load("fr_core_news_md")
except:
    raise RuntimeError(
        "Le modèle SpaCy 'fr_core_news_md' n'est pas installé. "
        "Veuillez l'installer en exécutant 'python -m spacy download fr_core_news_md' dans votre terminal."
    )

# Fonctions de traitement
def split_context_and_question(input_text: str):
    parts = re.split(r"Question:", input_text, maxsplit=1)

    if len(parts) == 2:
        context = re.sub(r"(Locuteur \d+:)", "", parts[0])
        question = re.sub(r"(Locuteur \d+:)", "", parts[1])
    else:
        context = ""
        question = re.sub(r"(Locuteur \d+:)", "", input_text)

    context = re.sub(r"#spk[\w-]*", "", context).strip()
    question = re.sub(r"#spk[\w-]*", "", question).strip()

    context = re.sub(r"\s+", " ", context).strip()
    question = re.sub(r"\s+", " ", question).strip()

    return context, question

# Analyse du corpus
def analyze_corpus(data):
    total_questions = len(data)
    label_counts = Counter(entry.get("label") for entry in data if "label" in entry)
    intention_counts = Counter(entry.get("intention") for entry in data if "intention" in entry)
    unique_intentions = len(intention_counts)

    return total_questions, label_counts, intention_counts, unique_intentions

# Vérification de la fréquence des labels
def freq_labels_ok(label_counts):
    if not label_counts:
        return True
    majority_count = max(label_counts.values())
    return all(count >= majority_count / 2 for count in label_counts.values())

# Normalisation des IDs
def normalize_ids(data):
    cleaned_data = []
    used_ids = set()
    next_id = 1

    for entry in data:
        raw_id = entry.get("id")
        try:
            new_id = int(raw_id)
        except:
            new_id = None
        if new_id is None or new_id in used_ids:
            while next_id in used_ids:
                next_id += 1
            new_id = next_id
            next_id += 1
        entry["id"] = new_id
        used_ids.add(new_id)
        cleaned_data.append(entry)
    return cleaned_data

# Nettoyage et lemmatisation
def clean_and_lemmatize(data):
    lemmas_count = []
    word_per_question = []
    pos_counter = Counter()

    for entry in data:
        question = entry.get("question", "")
        doc = nlp(question)
        lemmas_question = [t.lemma_ for t in doc if not t.is_punct and not t.is_space and not t.is_stop]
        tokens_question = [t.text for t in doc if not t.is_punct and not t.is_space]
        lemmas_count.extend(lemmas_question)
        word_per_question.append(len(tokens_question))

        for t in doc:
            if not t.is_punct and not t.is_space:
                pos_counter[t.pos_] += 1
        entry["lemmas"] = lemmas_question
        entry["nb_tokens"] = len(tokens_question)
    top_5_lemmas = Counter(lemmas_count).most_common(5)
    avg_words = sum(word_per_question)/len(word_per_question) if word_per_question else 0
    return {"top_5_lemmas": top_5_lemmas, "pos_distribution": dict(pos_counter), "avg_words_per_question": avg_words}

# Fusion des corpus
def fusion_corpus(data1, data2):
    return normalize_ids(data1 + data2)

# Création des graphiques
def plot_bar_chart(counter, title, xlabel, ylabel, filepath):
    if not counter:
        return
    
    items = sorted(counter.items(), key=lambda x:x[1], reverse=True)
    labels, values = zip(*items)
    plt.figure(figsize=(10,6))
    plt.bar(labels, values, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# Application principale
class StatsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyse de Corpus JSON")
        self.setGeometry(200, 200, 900, 600)

        # Barre de statut
        self.status_bar = QStatusBar()

        # Onglets
        self.tabs = QTabWidget()
        self.tab_load = QWidget()
        self.tab_stats = QWidget()
        self.tab_graphs = QWidget()
        self.tabs.addTab(self.tab_load, "Importation")
        self.tabs.addTab(self.tab_stats, "Statistiques")
        self.tabs.addTab(self.tab_graphs, "Graphiques")

        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)
        main_layout.addWidget(self.status_bar)
        self.setLayout(main_layout)

        # Initialisation des onglets
        self.init_tab_load()
        self.init_tab_stats()
        self.init_tab_graphs()

        # Données
        self.data1 = None
        self.data2 = None
        self.final_data = None

    # Onglet d'Importation
    def init_tab_load(self):
        layout = QVBoxLayout()
        self.tab_load.setLayout(layout)

        group = QGroupBox("Chargement du corpus JSON")
        g_layout = QVBoxLayout()
        group.setLayout(g_layout)

        self.load_button = QPushButton("Charger un fichier JSON")
        self.load_button.clicked.connect(self.load_first_json)
        g_layout.addWidget(self.load_button)

        self.file_info = QLabel("Aucun fichier chargé.")
        g_layout.addWidget(self.file_info)

        self.reset_button = QPushButton("Réinitialiser")
        self.reset_button.clicked.connect(self.reset_app)
        g_layout.addWidget(self.reset_button)

        layout.addWidget(group)

    # Onglet Statistiques
    def init_tab_stats(self):
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        layout = QVBoxLayout()
        layout.addWidget(self.stats_text)
        self.tab_stats.setLayout(layout)

    # Onglet Graphiques
    def init_tab_graphs(self):
        self.graphs_tabs = QTabWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.graphs_tabs)
        self.tab_graphs.setLayout(layout)

    # Nettoyage du layout
    def clear_graph_tabs(self):
        while self.graphs_tabs.count() > 0:
            widget = self.graphs_tabs.widget(0)
            self.graphs_tabs.removeTab(0)
            widget.deleteLater()

    # Chargement du premier JSON
    def load_first_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un fichier JSON", os.getcwd(), "Fichiers JSON (*.json)"
        )
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
        self.file_info.setText(f"Fichier chargé: {os.path.basename(file_path)} ({len(data)} entrées)")
        self.status_bar.showMessage("Premier fichier JSON chargé.", 5000)

        # Nettoyage automatique si input_text présent
        for entry in self.data1:
            if "input_text" in entry:
                context, question = split_context_and_question(entry["input_text"])
                entry["context"] = context
                entry["question"] = question
                del entry["input_text"]

        # Vérification des labels
        _, label_counts, _, _ = analyze_corpus(self.data1)
        if not freq_labels_ok(label_counts):
            self.ask_second_json()
        else:
            self.final_data = normalize_ids(self.data1)
            self.update_stats_and_graphs()

    # Second chargement JSON si nécessaire
    def ask_second_json(self):
        reply = QMessageBox.question(
            self, "Corpus incomplet",
            "Certaines classes sont sous-représentées.\nVoulez-vous charger un second corpus pour compléter ?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Ouvrir le second fichier JSON", os.getcwd(), "Fichiers JSON (*.json)"
            )
            if not file_path:
                self.final_data = normalize_ids(self.data1)
                self.update_stats_and_graphs()
                return
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data2 = json.load(f)
                if isinstance(data2, dict):
                    data2 = [data2]
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier JSON:\n{e}")
                self.final_data = normalize_ids(self.data1)
                self.update_stats_and_graphs()
                return
            self.data2 = data2

            # Nettoyage d'input_text
            for entry in self.data2:
                if "input_text" in entry:
                    context, question = split_context_and_question(entry["input_text"])
                    entry["context"] = context
                    entry["question"] = question
                    del entry["input_text"]

            # Fusion des corpus
            self.final_data = fusion_corpus(self.data1, self.data2)
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Enregistrer le corpus fusionné", "", "Fichier JSON (*.json)"
            )
            if save_path:
                if not save_path.lower().endswith(".json"):
                    save_path += ".json"
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.final_data, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, "Succès", f"Corpus fusionné sauvegardé :\n{save_path}")
            self.update_stats_and_graphs()
        else:
            self.final_data = normalize_ids(self.data1)
            self.update_stats_and_graphs()

    # Mise à jour des statistiques et graphiques
    def update_stats_and_graphs(self):
        if not self.final_data:
            return
        total_questions, label_counts, intention_counts, unique_intentions = analyze_corpus(self.final_data)
        stats_ling = clean_and_lemmatize(self.final_data)

        # Texte
        txt = f"Nombre total de questions : {total_questions}\n\n"
        txt += "Répartition des labels :\n"
        for l,c in label_counts.items():
            txt += f" - {l}: {c}\n"
        txt += "\nIntentions :\n"
        for i,c in intention_counts.items():
            txt += f" - {i}: {c}\n"
        txt += f"\nNombre d'intentions différentes : {unique_intentions}\n"
        txt += f"\nStatistiques linguistiques :\nMots moyens/question : {stats_ling['avg_words_per_question']:.2f}\n"
        txt += "Top 5 lemmes :\n"
        for lemma,freq in stats_ling["top_5_lemmas"]:
            txt += f" - {lemma}: {freq}\n"
        txt += "\nRépartition POS :\n"
        for pos,count in stats_ling["pos_distribution"].items():
            txt += f" - {pos}: {count}\n"

        self.stats_text.setText(txt)

        # Graphiques
        self.clear_graph_tabs()
        self.add_canvas_bar_chart(label_counts, "Labels")
        self.add_canvas_bar_chart(intention_counts, "Intentions")
        self.add_canvas_bar_chart(stats_ling["pos_distribution"], "POS")
        
        self.tabs.setCurrentWidget(self.tab_stats)
        self.status_bar.showMessage("Analyse terminée.", 5000)

        # Sauvegarde optionnelle
        reply = QMessageBox.question(
            self, "Sauvegarde",
            "Voulez-vous sauvegarder les graphiques et le résumé textuel ?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.save_graphs_and_resume(label_counts, intention_counts, stats_ling)

    # Ajout d'un graphique dans le layout
    def add_canvas_bar_chart(self, counter, title):
        if not counter:
            return
        
        items = sorted(counter.items(), key=lambda x:x[1], reverse=True)
        labels, values = zip(*items)

        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(labels, values, color='skyblue')
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        
        ax.set_title(title)
        ax.set_ylabel("Nombre")
        fig.tight_layout()
        
        canvas = FigureCanvas(fig)

        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        tab.setLayout(layout)

        self.graphs_tabs.addTab(tab, title)

    # Sauvegarde des graphiques et du résumé
    def save_graphs_and_resume(self, label_counts, intention_counts, stats_ling):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sauvegarde")
        if not folder:
            return
        try:
            plot_bar_chart(label_counts, "Fréquence des labels", "Labels", "Nombre", os.path.join(folder,"labels.png"))
            plot_bar_chart(intention_counts, "Fréquence des intentions", "Intentions", "Nombre", os.path.join(folder,"intentions.png"))
            plot_bar_chart(stats_ling["pos_distribution"], "Répartition POS", "POS", "Nombre", os.path.join(folder,"pos.png"))
            with open(os.path.join(folder,"resume_stats.txt"), 'w', encoding='utf-8') as f:
                f.write(self.stats_text.toPlainText())
            QMessageBox.information(self, "Succès", "Graphiques et résumé sauvegardés.")
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Erreur lors de la sauvegarde :\n{e}")

    # Réinitialisation de l'application
    def reset_app(self):
        self.data1 = None
        self.data2 = None
        self.final_data = None
        self.file_info.setText("Aucun fichier chargé.")
        self.stats_text.clear()
        self.clear_layout(self.graphs_layout)
        self.status_bar.showMessage("Application réinitialisée.", 5000)

# Fonction principale
def main():
    app = QApplication(sys.argv)
    window = StatsApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
