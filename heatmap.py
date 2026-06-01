import numpy as np
import pandas as pd
import keras
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from load_data_set import LoadDataSet

# -----------------------------
# 1. Modell laden
# -----------------------------
model_path = "models/modelTransferLearning7.keras"
model = keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": keras.applications.resnet_v2.preprocess_input
            }
        )

# -----------------------------
# 2. Testdaten laden
# -----------------------------
# Beispiel:
# X_test = np.load("X_test.npy")
# y_test = np.load("y_test.npy")

# Falls y_test one-hot encoded ist:
# y_test_labels = np.argmax(y_test, axis=1)

# Falls y_test bereits Klassenindices enthält:
# y_test_labels = y_test

# --- Beispielhafte Platzhalter ---
load_data = LoadDataSet(
    dataset_dir="train",
    dataset_test_dir="test"
)

test_dataset, class_names  = load_data.getTestDataset()

# -----------------------------
# 4. Vorhersagen berechnen
# -----------------------------
y_true = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)

y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

# -----------------------------
# 5. Confusion Matrix berechnen
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

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

display_class_names = [label_map.get(name, name) for name in class_names]

# -----------------------------
# 7. Heatmap plotten
# -----------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=display_class_names,
    yticklabels=display_class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Tatsächliche Klasse")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# -----------------------------
# 8. Als JPEG speichern
# -----------------------------
output_path = "confusion_matrix.jpg"
plt.savefig(output_path, format="jpg", dpi=300)
plt.show()

print(f"Confusion Matrix wurde gespeichert als: {output_path}")

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=display_class_names,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()
report_df = report_df.round(2)

plt.figure(figsize=(12, 8))
plt.axis("off")

table = plt.table(
    cellText=report_df.values,
    rowLabels=report_df.index,
    colLabels=report_df.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.2)

plt.title("Classification Report", pad=20)
plt.tight_layout()
plt.savefig("classification_report.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.close()