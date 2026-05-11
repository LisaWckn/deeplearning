import keras
from keras import layers, regularizers

def baseModel(inputShape):

    model = keras.Sequential()

    # Block 1
    model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='elu', input_shape=inputShape, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='elu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    model.add(keras.layers.Dropout(0.15))

    # Block 2
    model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='elu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='elu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    model.add(keras.layers.Dropout(0.15))

    # Block 3
    model.add(keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='elu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='elu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    model.add(keras.layers.Dropout(0.15))

    model.add(keras.layers.GlobalAveragePooling2D())

    # Dense Layers
    model.add(keras.layers.Dense(256, activation='elu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
    
    model.add(keras.layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))
    
    model.add(keras.layers.Dense(15, activation='softmax')) 
    
    return model

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.15),
])

# Normalisierung
normalization = keras.Sequential([
    layers.Rescaling(1./255),
])

# Dataset laden und aufteilen
full_dataset = keras.utils.image_dataset_from_directory(
    "train",
    image_size=(224, 224),
    batch_size=16,
    validation_split=0.2,
    subset="both",
    seed=123
)

train_dataset = full_dataset[0]
val_dataset = full_dataset[1]

# Normalisierung und Augmentation
train_dataset = train_dataset.map(lambda x, y: (normalization(x), y))
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
train_dataset = train_dataset.cache().shuffle(2000)

val_dataset = val_dataset.map(lambda x, y: (normalization(x), y))
val_dataset = val_dataset.cache()

model = baseModel((224, 224, 3))

# Adam Optimizer (Standard)
optimizer = keras.optimizers.Adam(learning_rate=0.001)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=8, 
    min_delta=0.005, 
    verbose=1, 
    restore_best_weights=True
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset, 
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping],
    verbose=1
)

model.save("model6.keras")