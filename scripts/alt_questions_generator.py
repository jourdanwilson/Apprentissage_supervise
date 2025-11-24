import sys
import json
import openai
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QSpinBox, QLineEdit,
    QPushButton, QTextEdit, QCheckBox, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt

class QuestionGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Générateur de Questions Alternatives")

        layout = QVBoxLayout()

        # Nombre de questions à générer
        layout.addWidget(QLabel("Nombre de questions à générer:"))
        self.nb_questions = QSpinBox()
        self.nb_questions.setMinimum(1)
        self.nb_questions.setMaximum(1000)
        layout.addWidget(self.nb_questions)

        # Clé API OpenAI
        layout.addWidget(QLabel("Clé API OpenAI:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.api_key_input)

        # Option pour inclure un contexte ou non
        self.include_context = QCheckBox("Générer également un contexte")
        layout.addWidget(self.include_context)

        # Bouton de génération
        self.generate_button = QPushButton("Générer")
        self.generate_button.clicked.connect(self.generate)
        layout.addWidget(self.generate_button)

        # Zone d'affichage des résultats
        layout.addWidget(QLabel("Aperçu des questions générées:"))
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        layout.addWidget(self.output_box)

        self.setLayout(layout)

    def generate(self):
        n = self.nb_questions.value()
        intention = "question canonique"
        api_key = self.api_key_input.text().strip()

        if not api_key:
            QMessageBox.warning(self, "Erreur", "Veuillez entrer votre clé API OpenAI.")
            return

        client = openai.OpenAI(api_key=api_key)

        # Prompt pour varier le vocabulaire des questions et générer le contexte
        prompt = f"Génère {n} questions alternatives de type 'tu préfères X ou Y'. "
        prompt += "Varie la formulation pour éviter que toutes les questions commencent par 'Tu préfères...'. "
        if self.include_context.isChecked():
            prompt += "Pour chaque question, génère également un contexte court de 1 à 2 phrases. "
        prompt += "Format exact : QUESTION: <q> CONTEXTE: <c>. Sépare chaque question par '---'."

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.choices[0].message.content
        except Exception as e:
            QMessageBox.critical(self, "Erreur API", str(e))
            return

        results = []
        txt_output = []

        # Découper et limiter à n questions exactes
        questions_raw = [q.strip() for q in text.split('---') if q.strip()][:n]

        for i, q_text in enumerate(questions_raw, start=1):
            q_id = str(i)  # IDs séquentiels (pour éviter les longs UUID)
            file_name = f"question_{i}"

            try:
                if "QUESTION:" in q_text and "CONTEXTE:" in q_text:
                    question = q_text.split("QUESTION:")[1].split("CONTEXTE:")[0].strip()
                    context = q_text.split("CONTEXTE:")[1].strip() if self.include_context.isChecked() else ""
                else:
                    question = q_text
                    context = ""
            except:
                question = q_text
                context = ""

            entry = {
                "id": q_id,
                "file_name": file_name,
                "label": "alternative",
                "intention": intention,
                "context": context,
                "question": question
            }

            results.append(entry)
            txt_output.append(f"ID: {q_id}\nFile: {file_name}\nQuestion: {question}\nContexte: {context}\n---\n")

        # Sauvegarde
        save_dir = QFileDialog.getExistingDirectory(self, "Choisir un dossier de sauvegarde")
        if not save_dir:
            return

        # Format TXT
        with open(f"{save_dir}/questions.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(txt_output))

        # Format JSON
        with open(f"{save_dir}/questions.json", "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=4, ensure_ascii=False)

        self.output_box.setText("\n".join(txt_output))
        QMessageBox.information(self, "Succès", "Les questions ont été générées et sauvegardées.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuestionGenerator()
    window.show()
    sys.exit(app.exec())