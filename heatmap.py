import numpy as np
import pandas as pd
import keras
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from load_data_set import LoadDataSet

MODEL_NAME = "modelFineTuned_Conv5Block2u3"
# Pfad zum gespeicherten Modell
model_path = f"models/{MODEL_NAME}.keras"

# Modell laden
# custom_objects wird benötigt, wenn beim Speichern / Laden
# modellbezogene Sonderfunktionen eingebunden sind
model = keras.models.load_model(
    model_path,
    custom_objects={
        "preprocess_input": keras.applications.resnet_v2.preprocess_input
    }
)

# Datensätze laden
load_data = LoadDataSet(
    dataset_dir="train",
    dataset_test_dir="test"
)

# Testdatensatz und Klassennamen abrufen
test_dataset, class_names  = load_data.getTestDataset()

# Echte Labels aus dem Testdatensatz sammeln
# Alle Labels aus den einzelnen Batches werden zu einem Array zusammengeführt
y_true = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)

# Vorhersagewahrscheinlichkeiten für alle Testdaten berechnen
y_pred_probs = model.predict(test_dataset)

# Aus den Wahrscheinlichkeiten jeweils die Klasse mit der höchsten Wahrscheinlichkeit wählen
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix berechnen
# Sie zeigt, welche Klassen korrekt bzw. falsch vorhergesagt wurden
cm = confusion_matrix(y_true, y_pred, normalize='true')

# Zuordnung interner Klassenbezeichnungen zu lesbaren Aktivitätsnamen
label_map = {
    "1": "using laptop",
    "2": "hugging",
    "3": "sleeping",
    "5": "clapping",
    "6": "dancing",
    "7": "cycling",
    "8": "calling",
    "10": "eating",
    "11": "fighting",
    "12": "listening to music"
}

# Anzeigenamen für die Achsen der Matrix erzeugen
display_class_names = [label_map.get(name, name) for name in class_names]

# Grafik für die Confusion Matrix erstellen
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,                    # Zellwerte anzeigen
    fmt=".2f",                     # 2 Nachkommastellen
    cmap="Blues",
    xticklabels=display_class_names,
    yticklabels=display_class_names
)

# Titel und Achsenbeschriftungen setzen
plt.title("Confusion Matrix")
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Tatsächliche Klasse")

# Achsenbeschriftungen besser lesbar drehen
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Confusion Matrix als Bild speichern
output_path = f"diagrams/{MODEL_NAME}/confusion_matrix.jpg"
plt.savefig(output_path, format="jpg", dpi=300)
plt.show()

print(f"Confusion Matrix wurde gespeichert als: {output_path}")

# Classification Report berechnen
# Enthält Precision, Recall, F1-Score und Support pro Klasse
report_dict = classification_report(
    y_true,
    y_pred,
    target_names=display_class_names,
    output_dict=True
)

# Report in ein DataFrame umwandeln und auf 2 Nachkommastellen runden
report_df = pd.DataFrame(report_dict).transpose()
report_df = report_df.round(2)

# Neue Grafik für die tabellarische Darstellung des Classification Reports
plt.figure(figsize=(12, 8))
plt.axis("off")  # Achsen ausblenden, da nur die Tabelle dargestellt werden soll

# Tabelle aus dem DataFrame erzeugen
table = plt.table(
    cellText=report_df.values,
    rowLabels=report_df.index,
    colLabels=report_df.columns,
    cellLoc="center",
    loc="center"
)

# Schriftgröße und Skalierung der Tabelle anpassen
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.2)

# Titel setzen und Darstellung optimieren
plt.title("Classification Report", pad=20)
plt.tight_layout()

# Classification Report als Bild speichern
report_path = f"diagrams/{MODEL_NAME}/classification_report.jpg"
plt.savefig(report_path, format="jpg", dpi=300, bbox_inches="tight")
plt.close()