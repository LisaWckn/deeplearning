import keras
from keras import layers
from keras.callbacks import TensorBoard
from datetime import datetime

from meilenstein2.models import Models
from load_data_set import LoadDataSet
from evaluate_test import EvaluateModel

# Anfangs-Lernrate:
# - Für Transfer Learning / nur Klassifikationskopf trainieren: 1e-3
# - Für Fine-Tuning des vortrainierten Modells: kleiner, 1e-4 bis 1e-6
INITIAL_LEARNING_RATE = 1e-4

# Gibt beim Start aus, welche Layer trainierbar sind
PRINT_LAYER_TRAINABLE = True

# Steuert, ob ein Fine-Tuning-Szenario aktiv ist
IS_FINETUNING = True

# Auswahl der Modellarchitektur aus der eigenen Models-Klasse
model_spec = Models((224,224,3)).modelFineTuned_Conv5Block2u3

# Laden der Trainings- und Testdaten aus den angegebenen Verzeichnissen
load_data = LoadDataSet(
    dataset_dir="train",
    dataset_test_dir="test"
)

# Trainings- und Validierungsdatensatz laden
train_dataset, val_dataset = load_data.getTrainDataset()

# Testdatensatz und die zugehörigen Klassennamen laden
test_dataset, class_names = load_data.getTestDataset()

# Sicherheitsprüfung:
# Labels in Train / Validation / Test müssen identisch sein,
# damit später korrekt trainiert und ausgewertet werden kann
if not load_data.check_label_consistency():
    raise ValueError("Label-Mismatch zwischen Train, Validation und Test festgestellt")

# Das eigentliche Keras-Modell aus der Modell-Spezifikation holen
model = model_spec.model

# Optional: Ausgabe, welche Layer trainierbar sind
if PRINT_LAYER_TRAINABLE:
    for layer in model.layers:
        print(f"{layer.name}: trainable={layer.trainable}")

    # Falls Fine-Tuning aktiv ist, zusätzlich alle Layer des Basis-Modells ausgeben
    if IS_FINETUNING:
        base_model = model.get_layer("resnet50v2")

        for layer in base_model.layers:
            print(layer.name, layer.trainable)

# Zusammenfassung der Modellarchitektur ausgeben
print(model.summary())

# TensorBoard-Callback:
# Speichert Trainingsverläufe, Histogramme und den Graphen für spätere Analyse
tensorboard_callback = TensorBoard(
    log_dir=f"logs/fit/{model_spec.name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    histogram_freq=1,      # Histogramme der Gewichte/Biases pro Epoche speichern
    write_graph=True,      # Modellgraph speichern
    write_images=True,     # Visualisierungen von Gewichten/Bildern erlauben
    update_freq='epoch'    # Logging einmal pro Epoche
)

# Early Stopping:
# Stoppt das Training, wenn sich die Validierungsgenauigkeit
# über mehrere Epochen nicht mehr verbessert
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=12,
    mode="max",
    restore_best_weights=True
)

# Reduziert die Lernrate automatisch,
# wenn sich der Validierungsfehler nicht weiter verbessert
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# Modell für das Training konfigurieren:
# - Adam als Optimierer
# - sparse_categorical_crossentropy für Integer-Labels
# - accuracy als Metrik zur Bewertung
model.compile(
    optimizer=keras.optimizers.Adam(INITIAL_LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training des Modells mit Trainings- und Validierungsdaten
history = model.fit(
    train_dataset, 
    validation_data=val_dataset,
    epochs=100,
    callbacks=[tensorboard_callback, early_stopping, reduce_lr]
)

# Speicherpfad für das trainierte Modell
model_path = f"models/{model_spec.name}.keras"

# Trainiertes Modell im Keras-Format speichern
model.save(model_path)

# Abschließende Auswertung auf dem Testdatensatz
EvaluateModel().evaluate_model(test_dataset, class_names, model_path)