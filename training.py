import keras
from keras import layers
from keras.callbacks import TensorBoard
from datetime import datetime

from models import Models
from load_data_set import LoadDataSet
from evaluate_test import EvaluateModel

model_spec = Models((224,224,3)).model6_normalized_augmented

load_data = LoadDataSet(
    dataset_dir="train",
    dataset_test_dir="test",
    normalize=model_spec.normalize,
    augment=model_spec.augment
)

train_dataset, val_dataset = load_data.getTrainDataset()
test_dataset, class_names = load_data.getTestDataset()

# Labels prüfen: Train / Validation / Test sollten identisch sein
if not load_data.check_label_consistency():
    raise ValueError("Label-Mismatch zwischen Train, Validation und Test festgestellt")

model = model_spec.model

# TensorBoard Callback für Logging
tensorboard_callback = TensorBoard(
    log_dir=f"logs/fit/{model_spec.name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    histogram_freq=1,  # Speichere Histogramme für Gewichte/Biases
    write_graph=True,  # Speichere Modellgraph
    write_images=True,  # Speichere Bilder (z.B. Filter)
    update_freq='epoch'  # Aktualisiere pro Epoche
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=12,
    mode="max",
    restore_best_weights=True
    )

reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset, 
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping, tensorboard_callback, reduce_lr]
)

model_path = f"models/{model_spec.name}.keras"

model.save(model_path)

EvaluateModel().evaluate_model(test_dataset, class_names, model_path)