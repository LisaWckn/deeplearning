import keras
from dataclasses import dataclass

@dataclass
class ModelSpec:
    model: keras.Model
    name: str
    normalize: bool = False
    augment: bool = False

class Models:

    def __init__(self, inputShape=(224,224,3)):
        self.model1 = self.getModel1(inputShape, model_name='model1')
        self.model1_normalized = self.getModel1(inputShape, model_name='model1_normalized', normalize=True)
        self.model1_augmented = self.getModel1(inputShape, model_name='model1_augmented', augment=True)
        self.model1_normalized_augmented = self.getModel1(inputShape, model_name='model1_normalized_augmented', normalize=True, augment=True)
        
        self.model2 = self.getModel2(inputShape, model_name='model2')
        self.model2_normalized = self.getModel2(inputShape, model_name='model2_normalized', normalize=True)
        self.model2_augmented = self.getModel2(inputShape, model_name='model2_augmented', augment=True)
        self.model2_normalized_augmented = self.getModel2(inputShape, model_name='model2_normalized_augmented', normalize=True, augment=True)

        self.model3 = self.getModel3(inputShape, model_name='model3')
        self.model3_normalized = self.getModel3(inputShape, model_name='model3_normalized', normalize=True)
        self.model3_augmented = self.getModel3(inputShape, model_name='model3_augmented', augment=True)
        self.model3_normalized_augmented = self.getModel3(inputShape, model_name='model3_normalized_augmented', normalize=True, augment=True)

        self.model4 = self.getModel4(inputShape, model_name='model4')
        self.model4_normalized = self.getModel4(inputShape, model_name='model4_normalized', normalize=True)
        self.model4_augmented = self.getModel4(inputShape, model_name='model4_augmented', augment=True)
        self.model4_normalized_augmented = self.getModel4(inputShape, model_name='model4_normalized_augmented', normalize=True, augment=True)

        self.model5 = self.getModel5(inputShape, model_name='model5')
        self.model5_normalized = self.getModel5(inputShape, model_name='model5_normalized', normalize=True)
        self.model5_augmented = self.getModel5(inputShape, model_name='model5_augmented', augment=True)
        self.model5_normalized_augmented = self.getModel5(inputShape, model_name='model5_normalized_augmented', normalize=True, augment=True)

    def getModel1(self, inputShape, model_name, normalize=False, augment=False):

        model = keras.Sequential()

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dense(units=15, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name, normalize=normalize, augment=augment)
    
    def getModel2(self, inputShape, model_name, normalize=False, augment=False):

        model = keras.Sequential()

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='elu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='elu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='elu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='elu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='elu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='elu'))
        model.add(keras.layers.Dense(units=15, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name, normalize=normalize, augment=augment)
    
    def getModel3(self, inputShape, model_name, normalize=False, augment=False):

        model = keras.Sequential()

        # Block 1: Mehr Conv-Schichten für tiefere Feature-Extraktion
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        # Block 2: Mehr Filter für komplexere Features
        model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        # Block 3: Noch mehr Filter für höhere Abstraktion
        model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())

        # Größere Dense-Schichten für bessere Klassifikation
        model.add(keras.layers.Dense(units=512, activation='relu'))
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dense(units=15, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name, normalize=normalize, augment=augment)
    
    def getModel4(self, inputShape, model_name, normalize=False, augment=False):

        model = keras.Sequential()

        # Block 1: Mehr Conv-Schichten für tiefere Feature-Extraktion
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=inputShape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(keras.layers.Dropout(0.25))

        # Block 2: Mehr Filter für komplexere Features
        model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(keras.layers.Dropout(0.25))

        # Block 3: Noch mehr Filter für höhere Abstraktion
        model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.GlobalAveragePooling2D())

        # Größere Dense-Schichten für bessere Klassifikation
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.Dense(units=15, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name, normalize=normalize, augment=augment)
    
    def getModel5(self, inputShape, model_name, normalize=False, augment=False):

        model = keras.Sequential()

        # Block 1: Mehr Conv-Schichten für tiefere Feature-Extraktion
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(keras.layers.Dropout(0.25))

        # Block 2: Mehr Filter für komplexere Features
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(keras.layers.Dropout(0.25))

        # Block 3: Noch mehr Filter für höhere Abstraktion
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.GlobalAveragePooling2D())

        # Größere Dense-Schichten für bessere Klassifikation
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=15, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name, normalize=normalize, augment=augment)
    
    def getModel6(self, inputShape, model_name, normalize=False, augment=False):

        model = keras.Sequential()

        # Block 1: Mehr Conv-Schichten für tiefere Feature-Extraktion
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(keras.layers.Dropout(0.25))

        # Block 2: Mehr Filter für komplexere Features
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(keras.layers.Dropout(0.25))

        # Block 3: Noch mehr Filter für höhere Abstraktion
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Conv2D(256, 3, padding="same", activation="relu"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(256, 3, padding="same", activation="relu"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D())
        model.add(keras.layers.Dropout(0.35))

        model.add(keras.layers.GlobalAveragePooling2D())

        # Größere Dense-Schichten für bessere Klassifikation
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=15, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name, normalize=normalize, augment=augment)