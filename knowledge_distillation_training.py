import keras
from keras import layers
from keras.callbacks import TensorBoard
from datetime import datetime
import tensorflow as tf

from distiller import Distiller
from evaluate_test import EvaluateModel
from load_data_set import LoadDataSet
from student_models import Models

num_classes = 10
input_shape = (224, 224, 3)

# Laden der Trainings- und Testdaten aus den angegebenen Verzeichnissen
load_data = LoadDataSet(
    dataset_dir="train",
    dataset_test_dir="test"
)

# Trainings- und Validierungsdatensatz laden
train_dataset, val_dataset = load_data.getTrainDataset()

# Testdatensatz und die zugehörigen Klassennamen laden
test_dataset, class_names = load_data.getTestDataset()

def build_teacher_logits_model():
    teacher_path = "meilenstein2/models/modelFineTuned_Conv5Block2u3.keras"
    teacher = keras.models.load_model(
            teacher_path,
            custom_objects={
                "preprocess_input": keras.applications.resnet_v2.preprocess_input
            }
        )
    
    penultimate_output = teacher.layers[-2].output
    last_dense = teacher.layers[-1]

    logits = keras.layers.Dense(
        units=last_dense.units,
        activation=None,
        name="teacher_logits"
    )(penultimate_output)

    teacher_logits_model = keras.Model(
        inputs=teacher.inputs,
        outputs=logits
    )

    teacher_logits_model.get_layer("teacher_logits").set_weights(
        last_dense.get_weights()
    )

    return teacher_logits_model

def build_teacher_without_aug(teacher, input_shape):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # ab preprocess_input weiterlaufen
    for layer in teacher.layers[5:]:
        x = layer(x)

    model = keras.Model(inputs, x, name="teacher_without_aug")

    return model

student_model_spec = Models((224,224,3)).student4_full
student = student_model_spec.model
print(student.summary())

# Total params: 24.782.986
teacher = build_teacher_logits_model()
if student_model_spec.augmented:
    teacher = build_teacher_without_aug(teacher, input_shape)
print(teacher.summary())
teacher.trainable = False

distiller = Distiller(student=student, teacher=teacher)

distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=student_model_spec.alpha,
    temperature=student_model_spec.temperature,
    augmented=student_model_spec.augmented
)

# TensorBoard-Callback:
# Speichert Trainingsverläufe, Histogramme und den Graphen für spätere Analyse
tensorboard_callback = TensorBoard(
    log_dir=f"logs/fit/{student_model_spec.name}/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    histogram_freq=1,      # Histogramme der Gewichte/Biases pro Epoche speichern
    write_graph=True,      # Modellgraph speichern
    write_images=True,     # Visualisierungen von Gewichten/Bildern erlauben
    update_freq='epoch'    # Logging einmal pro Epoche
)

# Early Stopping:
# Stoppt das Training, wenn sich die Validierungsgenauigkeit
# über mehrere Epochen nicht mehr verbessert
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_sparse_categorical_accuracy",
    patience=20,
    mode="max",
    restore_best_weights=True
)

# Reduziert die Lernrate automatisch,
# wenn sich der Validierungsfehler nicht weiter verbessert
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_student_loss",
    factor=0.5,
    patience=12,
    mode="min",
    min_lr=1e-6,
    verbose=1
)

distiller.fit(
    train_dataset, 
    validation_data=val_dataset,
    epochs=500,
    callbacks=[tensorboard_callback, early_stopping, reduce_lr],
    verbose=2
    )

student.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Speicherpfad für das trainierte Modell
model_path = f"models/{student_model_spec.name}.keras"

# Trainiertes Modell im Keras-Format speichern
student.save(model_path)

# Abschließende Auswertung auf dem Testdatensatz
EvaluateModel().evaluate_model(test_dataset, class_names, model_path)