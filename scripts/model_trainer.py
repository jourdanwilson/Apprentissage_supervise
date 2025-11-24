import sys
import os
import re
import json
import tempfile
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QCheckBox, QComboBox, QSpinBox, QTextEdit, QTabWidget, QGroupBox, QFormLayout, QProgressBar, QLineEdit)
from PySide6.QtGui import QPixmap

# Configuration de Seaborn pour les graphiques
sns.set_theme()

# Undersampler (sous-échantillonneur) personnalisé
def undersample_custom(df, label_col='label', targets=None, random_state=42):
    if targets is None:
        targets = {'totale': 150, 'partielle': 81, 'alternative': 81}

    groups = []
    for lab, g in df.groupby(label_col):
        target_size = targets.get(lab, len(g))
        if len(g) > target_size:
            groups.append(g.sample(target_size, random_state=random_state))
        else:
            groups.append(g)
    return pd.concat(groups).sample(frac=1, random_state=random_state).reset_index(drop=True)

class TextClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Entrainement de modèles (Types de questions)")
        self.resize(1000, 700)

        self.df = None
        self.results = {}
        self.models = {}
        self.tempdir = Path(tempfile.mkdtemp(prefix='trainer_'))

        # Layout principal
        main = QVBoxLayout()

        # Chargement des données
        load_box = QGroupBox('Chargement des données')
        load_layout = QFormLayout()

        self.load_btn = QPushButton("Charger JSON")
        self.load_btn.clicked.connect(self.load_json)
        load_layout.addRow(self.load_btn)

        self.text_choice = QComboBox()
        self.text_choice.addItems(['question', 'contexte + question'])
        load_layout.addRow(QLabel("Texte à utiliser:"), self.text_choice)

        undersample_box = QGroupBox('Sous-échantillonnage par classe')
        undersample_layout = QFormLayout()

        self.target_totale = QSpinBox()
        self.target_totale.setMinimum(1)
        self.target_totale.setMaximum(10000)
        self.target_totale.setValue(150)

        self.target_alternative = QSpinBox()
        self.target_alternative.setMinimum(1)
        self.target_alternative.setMaximum(10000)
        self.target_alternative.setValue(81)

        self.target_partielle = QSpinBox()
        self.target_partielle.setMinimum(1)
        self.target_partielle.setMaximum(10000)
        self.target_partielle.setValue(81)

        undersample_layout.addRow(QLabel("Totale:"), self.target_totale)
        undersample_layout.addRow(QLabel("Alternative:"), self.target_alternative)
        undersample_layout.addRow(QLabel("Partielle:"), self.target_partielle)

        undersample_box.setLayout(undersample_layout)
        load_layout.addRow(undersample_box)

        load_box.setLayout(load_layout)
        main.addWidget(load_box)

        # Sélection des classifieurs
        model_box = QGroupBox('Sélection des classifieurs')
        model_layout = QHBoxLayout()
        self.chk_svc = QCheckBox("LinearSVC")
        self.chk_lr = QCheckBox("LogisticRegression")
        self.chk_nb = QCheckBox("NaiveBayes")
        self.chk_rf = QCheckBox("RandomForest")

        self.chk_svc.setChecked(True)
        self.chk_lr.setChecked(True)
        self.chk_nb.setChecked(True)
        self.chk_rf.setChecked(True)

        model_layout.addWidget(self.chk_svc)
        model_layout.addWidget(self.chk_lr)
        model_layout.addWidget(self.chk_nb)
        model_layout.addWidget(self.chk_rf)

        model_box.setLayout(model_layout)
        main.addWidget(model_box)

        # Contrôles d'entrainement
        ctrl_layout = QHBoxLayout()
        self.train_btn = QPushButton("Lancer l'entrainement")
        self.train_btn.clicked.connect(self.run_training)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        ctrl_layout.addWidget(self.train_btn)
        ctrl_layout.addWidget(self.progress)
        main.addLayout(ctrl_layout)

        # Résultats
        results_box = QGroupBox('Résultats')
        results_layout = QVBoxLayout()

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        results_layout.addWidget(self.metrics_text)

        # Onglets pour les matrices de confusion
        self.tab_conf = QTabWidget()
        results_layout.addWidget(self.tab_conf)

        results_box.setLayout(results_layout)
        main.addWidget(results_box)

        # Options de sauvegarde
        save_box = QGroupBox('Sauvegarde des comparatifs et du meilleur modèle')
        save_layout = QFormLayout()
        self.save_folder_input = QLineEdit()
        browse_btn = QPushButton("Dossier de sauvegarde")
        browse_btn.clicked.connect(self.choose_save_folder)
        save_layout.addRow(browse_btn, self.save_folder_input)

        self.save_all_chk = QCheckBox("Sauvegarder tous les comparatifs (tables + matrices)")
        self.save_best_chk = QCheckBox("Sauvegarder le meilleur modèle")
        self.save_best_chk.setChecked(True)
        save_layout.addRow(self.save_all_chk)
        save_layout.addRow(self.save_best_chk)

        save_box.setLayout(save_layout)
        main.addWidget(save_box)

        # Bouton de réinitialisation
        self.reset_btn = QPushButton("Réinitialiser l'application")
        self.reset_btn.clicked.connect(self.reset_ui)
        ctrl_layout.addWidget(self.reset_btn)

        self.setLayout(main)

    # Choisir le dossier de sauvegarde
    def choose_save_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sauvegarde")
        if d:
            self.save_folder_input.setText(d)

    # Charger un fichier JSON
    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir le fichier JSON", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            self.df = pd.DataFrame(data)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier JSON:\n{e}")
            return
        
        # Assurer la présence des colonnes nécessaires
        for col in ['label', 'question', 'context', 'id']:
            if col not in self.df.columns:
                self.df[col] = ''

        # Nettoyer les IDs (si cela n'est pas déjà fait)
        def to_int_id(x):
            try:
                return int(x)
            except:
                return np.nan
        self.df['id'] = self.df['id'].apply(to_int_id)

        # Nettoyer les textes des balises #spk (si cela n'est pas déjà fait)
        self.df['context'] = self.df['context'].astype(str).apply(lambda t: re.sub(r'#spk[\w-]*', '', t))
        self.df['question'] = self.df['question'].astype(str).apply(lambda t: re.sub(r'#spk[\w-]*', '', t))

        QMessageBox.information(self, "Succès", f"Données chargées avec succès: {len(self.df)} entrées.")

    def run_training(self):
        if self.df is None:
            QMessageBox.warning(self, "Attention", "Veuillez d'abord charger un fichier JSON.")
            return
        
        # Préparer les données
        use = self.text_choice.currentText()
        if use == 'question':
            X = self.df['question'].astype(str)
        else:
            X = (self.df['context'].astype(str) + " " + self.df['question'].astype(str)).str.strip()
        y = self.df['label'].astype(str)

        # Sous-échantillonnage
        targets = {'totale': self.target_totale.value(), 'alternative': self.target_alternative.value(), 'partielle': self.target_partielle.value()}
        df_bal = undersample_custom(pd.DataFrame({'text': X, 'label': y}), label_col='label', targets=targets)
        Xb = df_bal['text']
        yb = df_bal['label']

        # Diviser en ensembles d'entrainement et de test
        X_train, X_test, y_train, y_test = train_test_split(Xb, yb, test_size=0.2, random_state=42, stratify=yb)

        # Sélection des classifieurs
        selected = []
        if self.chk_svc.isChecked():
            selected.append(('LinearSVC', LinearSVC())) 
        if self.chk_lr.isChecked():
            selected.append(('LogisticRegression', LogisticRegression(max_iter=1000)))
        if self.chk_nb.isChecked():
            selected.append(('NaiveBayes', MultinomialNB()))
        if self.chk_rf.isChecked():
            selected.append(('RandomForest', RandomForestClassifier(n_estimators=200)))

        if not selected:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner au moins un classifieur.")
            return
        
        self.metrics_text.clear()
        self.tab_conf.clear()
        self.results = {}
        self.models = {}

        total = len(selected)
        for idx, (name, clf) in enumerate(selected, start=1):
            self.progress.setValue(int(idx / total * 100))
            QApplication.processEvents()

            # Pipeline TF-IDF + Classifieur
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)),
                ('clf', clf)
            ])

            # Entrainement et évaluation
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            prec, rec, f1_per_class, sup = precision_recall_fscore_support(y_test, y_pred, average=None, labels=np.unique(y_test))

            # Détails par classe
            labels_uniques = np.unique(y_test)
            self.metrics_text.append(f"=== {name} - Détails par classe ===")
            for lab, p, r, f, s in zip(labels_uniques, prec, rec, f1_per_class, sup):
                self.metrics_text.append(
                    f" - {lab} :\n"
                    f"    Précision: {p:.4f}\n"
                    f"    Rappel: {r:.4f}\n"
                    f"    F1: {f:.4f}\n"
                    f"    Support: {s}\n\n"
                )

            # Stockage des résultats
            self.results[name] = {
                'y_test': list(y_test),
                'y_pred': list(y_pred),
                'accuracy': acc,
                'f1_macro': f1_macro,
            }
            self.models[name] = pipe

            # Affichage des résultats
            self.metrics_text.append(f"=== {name} ===")
            self.metrics_text.append(f"Précision: {acc:.4f} | F1 Macro: {f1_macro:.4f}\n")
            self.metrics_text.append(classification_report(y_test, y_pred))

            # Matrices de confusions
            labels = np.unique(y_test)
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', ax=ax)
            ax.set_xlabel('Prédit')
            ax.set_ylabel('Vrai')
            ax.set_title(f'Matrice de confusion - {name}')
            fig.tight_layout()
            img_path = self.tempdir / f'confusion_{name}.png'
            fig.savefig(img_path)
            plt.close(fig)

            # Ajouter l'onglet de la matrice de confusion
            lbl = QLabel()
            pix = QPixmap(str(img_path))
            lbl.setPixmap(pix)
            lbl.setScaledContents(True)
            self.tab_conf.addTab(lbl, name)

            # Graphique des scores F1 par classe
            fig_f1, ax_f1 = plt.subplots(figsize=(6,4))
            df_f1 = pd.DataFrame({'Classe': labels_uniques, 'F1': f1_per_class})
            sns.barplot(data=df_f1, x='Classe', y='F1', hue='Classe', dodge=False, palette='viridis', ax=ax_f1, legend=False)
            ax_f1.set_ylim(0, 1)
            ax_f1.set_title(f'Scores F1 par classe - {name}')
            ax_f1.set_xlabel('Classe')
            ax_f1.set_ylabel('Score F1')
            fig_f1.tight_layout()
            img_f1_path = self.tempdir / f'f1_scores_{name}.png'
            fig_f1.savefig(img_f1_path)
            plt.close(fig_f1)

            # Ajouter l'onglet des scores F1
            lbl_f1 = QLabel()
            pix_f1 = QPixmap(str(img_f1_path))
            lbl_f1.setPixmap(pix_f1)
            lbl_f1.setScaledContents(True)
            self.tab_conf.addTab(lbl_f1, f'F1 - {name}')

        self.progress.setValue(100)

        # Résumé des performances
        summary = []
        for name, info in self.results.items():
            summary.append({'modèle': name, 'précision': info['accuracy'], 'f1_macro': info['f1_macro']})
        df_summary = pd.DataFrame(summary).sort_values('f1_macro', ascending=False).reset_index(drop=True)

        self.metrics_text.append("=== Résumé des performances ===")
        self.metrics_text.append(df_summary.to_string(index=False))

        # Sauvegarde des résultats
        save_folder = self.save_folder_input.text().strip()
        if save_folder and (self.save_all_chk.isChecked() or self.save_best_chk.isChecked()):
            try:
                os.makedirs(save_folder, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Impossible de créer le dossier de sauvegarde:\n{e}")
                save_folder = ''

        if save_folder and self.save_all_chk.isChecked():

            # Sauvegarder le tableau comparatif
            df_summary.to_csv(os.path.join(save_folder, 'model_comparisons.csv'), sep=';', index=False, encoding='utf-8-sig')
            for name, info in self.results.items():

                # Sauvegarder le rapport de classification
                cr = classification_report(info['y_test'], info['y_pred'])
                with open(os.path.join(save_folder, f'classification_report_{name}.txt'), 'w', encoding='utf-8') as f:
                    f.write(cr)

                    # Sauvegarder les détails par classe
                    labels_uniques = np.unique(info['y_test'])
                    prec, rec, f1, sup = precision_recall_fscore_support(info['y_test'], info['y_pred'], average=None, labels=labels_uniques)

                    f.write(f"\n=== Détails par classe pour {name} ===\n")
                    for lab, p, r, f1c, s in zip(labels_uniques, prec, rec, f1, sup):
                        f.write(
                            f"    Classe: {lab}\n"
                            f"    Précision: {p:.4f}\n"
                            f"    Rappel: {r:.4f}\n"
                            f"    F1: {f1c:.4f}\n"
                            f"    Support: {s}\n\n"
                        )
                
                # Sauvegarder la matrice de confusion
                src = self.tempdir / f'confusion_{name}.png'
                if src.exists():
                    dst = Path(save_folder) / f'confusion_{name}.png'
                    with open(src, 'rb') as fr, open(dst, 'wb') as fw:
                        fw.write(fr.read())

                # Sauvegarder le graphique des scores F1
                img_f1_src = self.tempdir / f'f1_scores_{name}.png'
                if img_f1_src.exists():
                    img_f1_dst = Path(save_folder) / f'f1_scores_{name}.png'
                    with open(img_f1_src, 'rb') as fr, open(img_f1_dst, 'wb') as fw:
                        fw.write(fr.read())

        
        if save_folder and self.save_best_chk.isChecked() and not df_summary.empty:
            # Sauvegarder le meilleur modèle
            best_model_name = df_summary.loc[0, 'modèle']
            best_pipe = self.models[best_model_name]
            joblib.dump(best_pipe, os.path.join(save_folder, f'best_model_{best_model_name}.joblib'))
            QMessageBox.information(self, "Succès", f"Le meilleur modèle ({best_model_name}) a été sauvegardé avec succès.")

        QMessageBox.information(self, "Terminé", "L'entrainement est terminé.")

    def reset_ui(self):
        # Réinitialiser tous les champs de l'interface utilisateur
        self.text_choice.setCurrentIndex(0)
        self.target_totale.setValue(150)
        self.target_alternative.setValue(81)
        self.target_partielle.setValue(81)
        self.chk_svc.setChecked(True)
        self.chk_lr.setChecked(True)
        self.chk_nb.setChecked(True)
        self.chk_rf.setChecked(True)
        self.save_folder_input.clear()
        self.save_all_chk.setChecked(False)
        self.save_best_chk.setChecked(True)

        # Effacer les résultats affichés
        self.metrics_text.clear()
        self.tab_conf.clear()

        # Réinitialiser la barre de progression
        self.progress.setValue(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextClassifierApp()
    window.show()
    sys.exit(app.exec())

            











            
        