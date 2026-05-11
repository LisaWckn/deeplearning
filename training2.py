import keras
from keras import layers

def baseModel(inputShape):

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=inputShape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(units=15, activation='softmax')) 
    
    return model

# Data Augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Dataset laden und aufteilen
full_dataset = keras.utils.image_dataset_from_directory(
    "train",
    image_size=(224, 224),
    batch_size=32,
    validation_split=0.2,
    subset="both",
    seed=123
)

train_dataset = full_dataset[0].cache().shuffle(1000)
val_dataset = full_dataset[1].cache()

# Augmentation auf Trainingsdaten anwenden
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

model : keras.Sequential = baseModel((224,224,3))

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, verbose=1, restore_best_weights=True)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset, 
    validation_data=val_dataset,
    epochs=50,
    callbacks=[early_stopping]
)

model.save("optimized_model.keras")