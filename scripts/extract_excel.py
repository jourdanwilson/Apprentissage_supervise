import sys
import json
import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QLabel, QMessageBox, QHeaderView, QDialog, QTextEdit
)
from PySide6.QtCore import Qt

# Configuration des locuteurs et leurs tags
SPEAKER_MAP = {
    "#spk1:": "Locuteur 1",
    "#spk2:": "Locuteur 2"
}

# Fonction pour traiter et reformater le texte
def process_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    lines = text.splitlines()
    processed_lines = []

    for line in lines:
        matched = False
        for tag, speaker in SPEAKER_MAP.items():
            if line.startswith(tag):
                content = line[len(tag):].strip()
                if content.startswith(f"{speaker}:"):
                    processed_lines.append(content)
                else:
                    processed_lines.append(f"{speaker}: {content}")
                matched = True
                break
        if not matched and line.strip():
            processed_lines.append(line.strip())

    return " ".join(processed_lines)

# Fenêtre de prévisualisation JSON
class JsonPreview(QDialog):
    def __init__(self, json_text):
        super().__init__()
        self.setWindowTitle("Prévisualisation du JSON")
        self.resize(700, 600)

        layout = QVBoxLayout(self)
        text_widget = QTextEdit()
        text_widget.setReadOnly(True)
        text_widget.setPlainText(json_text)

        layout.addWidget(text_widget)

# Classe principale de l'application
class ExcelToJson(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Excel ➜ JSON")
        self.resize(1100, 700)

        # Layout principal
        main_layout = QVBoxLayout(self)

        # En-tête
        header = QLabel("<h2>Convertisseur Excel ➜ JSON</h2>")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        # Barre d'outils
        toolbar_layout = QHBoxLayout()

        self.load_btn = QPushButton("Ouvrir Excel")
        self.load_btn.clicked.connect(self.load_excel)
        toolbar_layout.addWidget(self.load_btn)

        self.export_btn = QPushButton("Export JSON")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_json)
        toolbar_layout.addWidget(self.export_btn)

        self.preview_btn = QPushButton("Prévisualiser JSON")
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self.preview_json)
        toolbar_layout.addWidget(self.preview_btn)

        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        # Label d'information
        self.info_label = QLabel("Aucun fichier chargé.")
        main_layout.addWidget(self.info_label)

        # Tableau pour afficher les données Excel
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        main_layout.addWidget(self.table)

        self.df = None
        self.last_json_data = None

    # Charger un fichier Excel
    def load_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un fichier Excel", "", "Fichiers Excel (*.xlsx *.xls)"
        )
        if not file_path:
            return

        try:
            df = pd.read_excel(file_path, header=0)
            if "id" in df.columns:
                df = df.sort_values(by="id", ascending=True)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger le fichier :\n{e}")
            return

        # Nettoyage des colonnes
        df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)
        df = df.loc[:, ~df.columns.str.match(r'^Unnamed: \d+$')]

        if df.index.name and df.index.name.lower() == "id":
            df.reset_index(inplace=True)

        # Traitement des champs
        for col in ["previous_context", "question"]:
            if col in df.columns:
                df[col] = df[col].apply(process_text)

        self.df = df
        self.info_label.setText(f"Fichier chargé : {file_path} ({len(df)} lignes)")
        self.populate_table()

        self.export_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)

    # Remplir le tableau avec les données du DataFrame
    def populate_table(self):
        df = self.df
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels(df.columns)

        for r in range(len(df)):
            for c in range(len(df.columns)):
                val = df.iat[r, c]
                val = "" if pd.isna(val) else str(val)
                self.table.setItem(r, c, QTableWidgetItem(val))

    # Création des données JSON
    def build_json_data(self):
        df = self.df
        safe_cols = {col.lower().strip(): col for col in df.columns}

        json_list = []
        for idx, row in df.iterrows():
            prev_ctxt = row.get(safe_cols.get("previous_context"), "")
            question = row.get(safe_cols.get("question"), "")

            item = {
                "id": row.get(safe_cols.get("id"), idx + 1),
                "file_name": row.get(safe_cols.get("file_name")),
                "input_text": f"{prev_ctxt} Question: {question}".strip(),
                "label": row.get(safe_cols.get("type de question")),
                "intention": row.get(safe_cols.get("intention"))
            }
            json_list.append(item)

        return json_list

    # Prévisualiser le JSON
    def preview_json(self):
        if self.df is None:
            return

        data = self.build_json_data()
        json_text = json.dumps(data, ensure_ascii=False, indent=2)

        dlg = JsonPreview(json_text)
        dlg.exec()

    # Exporter le JSON
    def export_json(self):
        if self.df is None:
            return

        data = self.build_json_data()
        json_text = json.dumps(data, ensure_ascii=False, indent=2)

        json_path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer JSON", "", "Fichier JSON (*.json)"
        )
        if not json_path:
            return
        if not json_path.endswith(".json"):
            json_path += ".json"

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json_text)
            QMessageBox.information(self, "Succès", "JSON exporté avec succès !")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible d'enregistrer le fichier :\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ExcelToJson()
    viewer.show()
    sys.exit(app.exec())
