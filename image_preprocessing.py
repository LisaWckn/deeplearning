from pathlib import Path
import pandas as pd
import shutil

# Pfade
csv_file = "training_set.csv"
source_dir = Path("train")
target_dir = Path("train_labeled_directory")

# CSV laden
df = pd.read_csv(csv_file)

# Erwartete Struktur:
# filename,label

for _, row in df.iterrows():

    filename = row["filename"]
    label = row["label"]

    # Zielordner für Klasse
    class_dir = target_dir / label
    class_dir.mkdir(parents=True, exist_ok=True)

    # Quelldatei
    src = source_dir / filename

    # Zieldatei
    dst = class_dir / filename

    # Datei kopieren
    shutil.copy2(src, dst)

    print(f"{filename} -> {label}/")