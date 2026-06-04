import keras
from keras import layers
from keras.callbacks import TensorBoard
from datetime import datetime

from differential_lr_model import DifferentialLRModel
from models import Models
from load_data_set import LoadDataSet
from evaluate_test import EvaluateModel

# Variable zur Anpassung der Lernrate
# TransferLearning und Trainieren des Klassifikationskopfes: 1e-3
# Finetuning: 1e-5
INITIAL_LEARNING_RATE = 1e-4

model_spec = Models((224,224,3)).modelFineTuned_10Layer

load_data = LoadDataSet(
    dataset_dir="train",
    dataset_test_dir="test"
)

train_dataset, val_dataset = load_data.getTrainDataset()
test_dataset, class_names = load_data.getTestDataset()

# Labels prüfen: Train / Validation / Test sollten identisch sein
if not load_data.check_label_consistency():
    raise ValueError("Label-Mismatch zwischen Train, Validation und Test festgestellt")

model = model_spec.model

for layer in model.layers:
    print(f"{layer.name}: trainable={layer.trainable}")

base_model = model.get_layer("resnet50v2")

for layer in base_model.layers:
    print(layer.name, layer.trainable)

print(model.summary())

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

'''
diff_model = DifferentialLRModel(model, backbone_layer_name="resnet50v2")

diff_model.compile(
    backbone_optimizer=keras.optimizers.Adam(learning_rate=1e-6),
    head_optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss_fn=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]
)

history = diff_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[tensorboard_callback, early_stopping]
)'''

model.compile(
    optimizer=keras.optimizers.Adam(INITIAL_LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset, 
    validation_data=val_dataset,
    epochs=100,
    callbacks=[tensorboard_callback, early_stopping, reduce_lr]
)

model_path = f"models/{model_spec.name}.keras"

model.save(model_path)

EvaluateModel().evaluate_model(test_dataset, class_names, model_path)