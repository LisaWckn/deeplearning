import keras
from keras import layers
from dataclasses import dataclass

@dataclass
class ModelSpec:
    model: keras.Model
    name: str
    augmented: bool
    alpha: float
    temperature: float

class Models:

    def __init__(self, inputShape=(224,224,3)):
        # Trainable params: 24.202 Params
        # Standard: Test 51.40%, Train 64.87%, Val 53,21%, Epochen 127
        # Augmentiert: Test 42.76%, Train 50.30%, Val 43.93%, Epochen 67
        # Normalisiert: Test 44.67%, Train 51.71%, Val 46.85%, Epochen 118
        self.student1 = self.getStudent1(inputShape, model_name='student1', augmented=False)
        self.student1_augmented = self.getStudent1(inputShape, model_name='student1_augmented', augmented=True)
        self.student1_normalized = self.getStudent1(inputShape, model_name='student1_normalized', augmented=False, normalize=True)

        # Trainable params: 33.450 Params
        # Standard: Test 51.71%, Train 67.74%, Val 54.64%, Epochen 93
        # Augmentiert: Test 47.17%, Train 57.49%, Val 47.26%, Epochen 76
        # Normalisiert: Test 46.77%, Train 55.09%, Val 48.69%, Epochen 134
        self.student2 = self.getStudent2(inputShape, model_name='student2', augmented=False)
        self.student2_augmented = self.getStudent2(inputShape, model_name='student2_augmented', augmented=True)
        self.student2_normalized = self.getStudent2(inputShape, model_name='student2_normalized', augmented=False, normalize=True)

        # Trainable params: 84.426 Params
        # Standard: Test 55.95%, Train 81.83%, Val 54.76%, Epochen 67
        # Augmentiert: Test 53.45%, Train 71.26%, Val 54.35%, Epochen 109
        # Normalisiert: Test 54.10%, Train 73.84%, Val 54.46%, Epochen 91
        self.student3 = self.getStudent3(inputShape, model_name='student3', augmented=False)
        self.student3_augmented = self.getStudent3(inputShape, model_name='student3_augmented', augmented=True)
        self.student3_normalized = self.getStudent3(inputShape, model_name='student3_normalized', augmented=False, normalize=True)

        # Total params: 1.286.986 Params
        # Trainable params: 1.284.234 Params
        # Standard: Test 64.20%, Train 99.40%, Val 65.23%, Epochen 81
        # Augmentiert: Test 69.05%, Train 96.52%, Val 70.71%, Epochen 89
        # Normalisiert: Test 65.82%, Train 99.48%, Val 66.01%, Epochen 85
        self.student4 = self.getStudent4(inputShape, model_name='student4', augmented=False)
        self.student4_augmented = self.getStudent4(inputShape, model_name='student4_augmented', augmented=True)
        self.student4_normalized = self.getStudent4(inputShape, model_name='student4_normalized', augmented=False, normalize=True)
        
        # Alpha=0.5: Test 68.55%, Train 96.43%, Val 70.42%, Epochen 107
        # Alpha=0.9: Test 68.35%, Train 97.51%, Val 69.40%, Epochen107
        self.student4_augmented_alpha05 = self.getStudent4(inputShape, model_name='student4_augmented_alpha05', augmented=True, alpha=0.5)
        self.student4_augmented_alpha09 = self.getStudent4(inputShape, model_name='student4_augmented_alpha09', augmented=True, alpha=0.9)
        
        # Temperature=2.0: Test 69.47%, Train 96.59%, Val 69,11%, Epochen 83
        # Temperature=4.0: Test 70.68%, Train 97.72%, Val 72,86%, Epochen 112
        # Temperature=5.0: Test 70.76%, Train 95.45%, Val 71.13%, Epochen 97
        # Temperature=6.0: Test 70.37%, Train 94.24%, Val 71.07%, Epochen 71
        self.student4_augmented_temperature2 = self.getStudent4(inputShape, model_name='student4_augmented_temperature2', augmented=True, temperature=2.0)
        self.student4_augmented_temperature4 = self.getStudent4(inputShape, model_name='student4_augmented_temperature4', augmented=True, temperature=4.0)
        self.student4_augmented_temperature5 = self.getStudent4(inputShape, model_name='student4_augmented_temperature5', augmented=True, temperature=5.0)
        self.student4_augmented_temperature6 = self.getStudent4(inputShape, model_name='student4_augmented_temperature6', augmented=True, temperature=6.0)

        # Student4 Full: Test 71.55%, Train 96.58%, Val 72.32%, Epochen 76
        self.student4_full = self.getStudent4(inputShape, model_name='student4_full', augmented=True, normalize=True, alpha=0.5, temperature=5.0)

        # Total params: 2.560.298 Params
        # Trainable params: 2.555.818 Params
        # Standard: Test 63.69%, Train 99.49%, Val 64.64%, Epochen 82
        # Augmentiert: Test 67.48%, Train 93.02%, Val 70.77%, Epochen 80
        # Normalisiert: Test 65.52%, Train 99.48%, Val 66.55%, Epochen 84
        self.student5 = self.getStudent5(inputShape, model_name='student5', augmented=False)
        self.student5_augmented = self.getStudent5(inputShape, model_name='student5_augmented', augmented=True)
        self.student5_normalized = self.getStudent5(inputShape, model_name='student5_normalized', augmented=False, normalize=True)

        # Alpha=0.5: Test 70.65%, Train 98.29%, Val 71.89%, Epochen 101
        # Alpha=0.9: Test 69.05%, Train 96.18%, Val 68.04%, Epochen 98
        self.student5_augmented_alpha05 = self.getStudent5(inputShape, model_name='student5_augmented_alpha05', augmented=True, alpha=0.5)
        self.student5_augmented_alpha09 = self.getStudent5(inputShape, model_name='student5_augmented_alpha09', augmented=True, alpha=0.9)

        # Temperature=2.0: Test 67.76%, Train 95.45%, Val 69.17%, Epochen 77
        # Temperature=4.0: Test 69.95%, Train 96.34%, Val 71.55%, Epochen 101
        # Temperature=5.0: Test 69.36%, Train 93.14%, Val 69.94%, Epochen 94
        # Temperature=6.0: Test 72.56%, Train 97.49%, Val 73.21%, Epochen 99
        self.student5_augmented_temperature2 = self.getStudent5(inputShape, model_name='student5_augmented_temperature2', augmented=True, temperature=2.0)
        self.student5_augmented_temperature4 = self.getStudent5(inputShape, model_name='student5_augmented_temperature4', augmented=True, temperature=4.0)
        self.student5_augmented_temperature5 = self.getStudent5(inputShape, model_name='student5_augmented_temperature5', augmented=True, temperature=5.0)
        self.student5_augmented_temperature6 = self.getStudent5(inputShape, model_name='student5_augmented_temperature6', augmented=True, temperature=6.0)

        # Student5 Full: Test 70.17%, Train 95.89%, Val 69.70%, Epochen 84
        self.student5_full = self.getStudent5(inputShape, model_name='student5_full', augmented=True, normalize=True, alpha=0.5, temperature=6.0)

    def getStudent1(self, 
                    inputShape, 
                    model_name, 
                    augmented, 
                    alpha=0.1, 
                    temperature=3.0, 
                    normalize=False):
        # Trainable params: 24.202 Params

        model = keras.Sequential()

        model.add(layers.Input(shape=inputShape))

        if normalize:
            model.add(keras.layers.Lambda(
                keras.applications.resnet_v2.preprocess_input
            ))

        model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(layers.MaxPooling2D())

        model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))

        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=10)) 
        
        return ModelSpec(model=model, name=model_name, augmented=augmented, alpha=alpha, temperature=temperature)
    
    def getStudent2(self, 
                    inputShape, 
                    model_name, 
                    augmented, 
                    alpha=0.1, 
                    temperature=3.0, 
                    normalize=False):
        # Trainable params: 33.450 Params

        model = keras.Sequential()

        model.add(layers.Input(shape=inputShape))

        if normalize:
            model.add(keras.layers.Lambda(
                keras.applications.resnet_v2.preprocess_input
            ))

        model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(layers.MaxPooling2D())

        model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))

        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=10)) 
        
        return ModelSpec(model=model, name=model_name, augmented=augmented, alpha=alpha, temperature=temperature)
    
    def getStudent3(self, 
                    inputShape, 
                    model_name, 
                    augmented, 
                    alpha=0.1, 
                    temperature=3.0, 
                    normalize=False):
        # Trainable params: 84.426 Params

        model = keras.Sequential()

        model.add(layers.Input(shape=inputShape))

        if normalize:
            model.add(keras.layers.Lambda(
                keras.applications.resnet_v2.preprocess_input
            ))

        model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(layers.MaxPooling2D())

        model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(layers.MaxPooling2D())

        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(units=128, activation='relu'))
        model.add(layers.Dense(units=10)) 
        
        return ModelSpec(model=model, name=model_name, augmented=augmented, alpha=alpha, temperature=temperature)
    
    def getStudent4(self, 
                    inputShape, 
                    model_name, 
                    augmented, 
                    alpha=0.1, 
                    temperature=3.0, 
                    normalize=False):
        # Total params: 1.286.986 Params
        # Trainable params: 1.284.234 Params

        model = keras.Sequential()

        model.add(layers.Input(shape=inputShape))

        if normalize:
            model.add(keras.layers.Lambda(
                keras.applications.resnet_v2.preprocess_input
            ))

        model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(layers.GlobalAveragePooling2D())

        model.add(layers.Dense(units=256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(units=128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(units=10)) 

        return ModelSpec(model=model, name=model_name, augmented=augmented, alpha=alpha, temperature=temperature)
    
    def getStudent5(self, 
                    inputShape, 
                    model_name, 
                    augmented, 
                    alpha=0.1, 
                    temperature=3.0, 
                    normalize=False):
        # Total params: 2,560,298 (9.77 MB)
        # Trainable params: 2,555,818 (9.75 MB)

        model = keras.Sequential()

        model.add(layers.Input(shape=inputShape))

        if normalize:
            model.add(keras.layers.Lambda(
                keras.applications.resnet_v2.preprocess_input
            ))

        # Block 1
        model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))

        # Block 2
        model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))

        # Block 3 - etwas tiefer
        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))

        # Block 4 - etwas breiter und tiefer
        model.add(layers.Conv2D(288, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(288, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(288, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2))

        model.add(layers.GlobalAveragePooling2D())

        # Head - moderat verstärkt
        model.add(layers.Dense(units=512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Dense(units=256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))

        model.add(layers.Dense(units=10))

        return ModelSpec(model=model, name=model_name, augmented=augmented, alpha=alpha, temperature=temperature)