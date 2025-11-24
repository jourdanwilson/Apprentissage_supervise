import sys
import pandas as pd
import json
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QLabel

# Liste des locuteurs et leurs tags correspondants
SPEAKER_MAP = {
    "#spk1:": "Locuteur 1",
    "#spk2:": "Locuteur 2"
}

# Fonction pour traiter le texte et reformater les lignes
def process_text(text):
    if not isinstance(text, str):
        return ""
    
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

# Classe principale de l'application
class ExcelToJson(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel vers JSON")
        self.setGeometry(200, 200, 900, 600)

        self.layout = QVBoxLayout(self)

        self.info_label = QLabel("Aucun fichier chargé.")
        self.layout.addWidget(self.info_label)

        self.load_button = QPushButton("Charger un fichier Excel")
        self.load_button.clicked.connect(self.load_excel)
        self.layout.addWidget(self.load_button)

        self.export_button = QPushButton("Exporter en JSON")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_json)
        self.layout.addWidget(self.export_button)
        
        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        self.df = None
        
    # Méthode pour charger le fichier Excel
    def load_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Ouvrir un fichier Excel", "", "Fichiers Excel (*.xlsx *.xls)")
        if not file_path:
            return

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            self.info_label.setText(f"Erreur lors du chargement du fichier: {e}")
            return
        
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if df.index.name == 'id':
            df.reset_index(inplace=True)
        
        if 'previous_context' in df.columns:
            df['previous_context'] = df['previous_context'].apply(process_text)
        if 'question' in df.columns:
            df['question'] = df['question'].apply(process_text)

        self.df = df
        self.info_label.setText(f"Fichier chargé: {file_path}")
        self.populate_table()
        self.export_button.setEnabled(True)

    # Méthode pour remplir la table avec les données du DataFrame
    def populate_table(self):
        if self.df is None:
            return
        
        df = self.df
        self.table.setColumnCount(len(df.columns))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels(df.columns)

        for row_idx in range(len(df)):
            for col_idx in range(len(df.columns)):
                value = str(df.iat[row_idx, col_idx])
                self.table.setItem(row_idx, col_idx, QTableWidgetItem(value))

    # Méthode pour exporter les données en JSON
    def export_json(self):
        if self.df is None:
            self.info_label.setText("Aucun fichier chargé à exporter.")
            return
        
        specialized_data = []
        for index, row in self.df.iterrows():
            prev_ctxt = row.get('previous_context', "")
            question = row.get('question', "")
            input_text = f"{prev_ctxt} Question: {question}".strip()

            item = {
                "id": row.get("id", index + 1),
                "file_name": row.get("file_name"),
                "input_text": input_text,
                "label": row.get("type de question"),
                "intention": row.get("Intention")
            }
            specialized_data.append(item)

        json_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le fichier JSON", "", "Fichiers JSON (*.json)")
        if not json_path:
            return
        
        if not json_path.lower().endswith('.json'):
            json_path += '.json'

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(specialized_data, f, ensure_ascii=False, indent=2)
            self.info_label.setText(f"Fichier JSON exporté avec succès: {json_path}")
        except Exception as e:
            self.info_label.setText(f"Erreur lors de l'exportation du fichier JSON: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ExcelToJson()
    viewer.show()
    sys.exit(app.exec())
