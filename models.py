import keras
from dataclasses import dataclass

@dataclass
class ModelSpec:
    model: keras.Model
    name: str

class Models:

    def __init__(self, inputShape=(224,224,3)):

        self.modelTransferLearning1 = self.getTransferLearningModelResNet50V2(inputShape, model_name='modelTransferLearning1')

        self.modelTransferLearning2 = self.getTransferLearningModelResNet50V2_2(inputShape, model_name='modelTransferLearning2')

        self.modelTransferLearning3 = self.getTransferLearningModelResNet50V2_3(inputShape, model_name='modelTransferLearning3')

        self.modelTransferLearning4 = self.getTransferLearningModelResNet50V2_4(inputShape, model_name='modelTransferLearning4')

        self.modelTransferLearning5 = self.getTransferLearningModelResNet50V2_5(inputShape, model_name='modelTransferLearning5')

        self.modelTransferLearning6 = self.getTransferLearningModelResNet50V2_6(inputShape, model_name='modelTransferLearning6')

        self.modelTransferLearning7 = self.getTransferLearningModelResNet50V2_7(inputShape, model_name='modelTransferLearning7')

        self.modelTransferLearning8 = self.getTransferLearningModelResNet50V2_8(inputShape, model_name='modelTransferLearning8')

        # Bis hier hin bestes Modell: modelTransferLearning7 mit 78.17%
        # Modell modelTransferLearning7 wird finegetuned
        # Anzahl der zum Training hinzugenommen Layer aus dem ResNet50 Modell werden schrittweise erhöht

        self.modelFineTuned_5Layer = self.fineTuneResNet50V2("models/modelTransferLearning7.keras", 5, 'modelFineTuned_5Layer') #77.95%, 77.95%
        self.modelFineTuned_10Layer = self.fineTuneResNet50V2("models/modelTransferLearning7.keras", 10, 'modelFineTuned_10Layer') #78.09%, 78.06%
        self.modelFineTuned_20Layer = self.fineTuneResNet50V2("models/modelTransferLearning7.keras", 20, 'modelFineTuned_20Layer') # 78.20%
        self.modelFineTuned_30Layer = self.fineTuneResNet50V2("models/modelTransferLearning7.keras", 30, 'modelFineTuned_30Layer') # 77.95%

    def getTransferLearningModelResNet50V2(self, inputShape, model_name):
        # Basismodell für den Klassifikationskopf
        # 76.43%

        resNet50V2 = keras.applications.ResNet50V2(include_top=False, 
                                                   weights="imagenet",
                                                   input_shape=inputShape)

        for layer in resNet50V2.layers:
            layer.trainable = False

        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=inputShape))

        model.add(keras.layers.Lambda(
            keras.applications.resnet_v2.preprocess_input
        ))
        model.add(resNet50V2)

        model.add(keras.layers.GlobalAveragePooling2D())
        model.add(keras.layers.Dense(units=10, activation='softmax'))

        return ModelSpec(model=model, name=model_name)
    
    def getTransferLearningModelResNet50V2_2(self, inputShape, model_name):
        # 76.74%
        # Gleiche Genauigkeit, weniger Overfitting

        resNet50V2 = keras.applications.ResNet50V2(include_top=False, 
                                                   weights="imagenet",
                                                   input_shape=inputShape)

        for layer in resNet50V2.layers:
            layer.trainable = False

        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=inputShape))

        # Augmentation hinzugefügt
        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.1))
        model.add(keras.layers.RandomZoom(0.1))
        model.add(keras.layers.RandomContrast(0.2))

        model.add(keras.layers.Lambda(
            keras.applications.resnet_v2.preprocess_input
        ))
        model.add(resNet50V2)

        model.add(keras.layers.GlobalAveragePooling2D())
        model.add(keras.layers.Dense(units=10, activation='softmax'))

        return ModelSpec(model=model, name=model_name)
    
    def getTransferLearningModelResNet50V2_3(self, inputShape, model_name):
        # 77.69%
        # Wieder mehr Overfitting, aber auch etwas höhere Qualität

        resNet50V2 = keras.applications.ResNet50V2(include_top=False, 
                                                   weights="imagenet",
                                                   input_shape=inputShape)

        for layer in resNet50V2.layers:
            layer.trainable = False

        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=inputShape))

        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.1))
        model.add(keras.layers.RandomZoom(0.1))
        model.add(keras.layers.RandomContrast(0.2))

        model.add(keras.layers.Lambda(
            keras.applications.resnet_v2.preprocess_input
        ))
        model.add(resNet50V2)

        model.add(keras.layers.GlobalAveragePooling2D())

        # Weiteren Dense-Layer hinzugefügt
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dense(units=10, activation='softmax'))

        return ModelSpec(model=model, name=model_name)
    
    def getTransferLearningModelResNet50V2_4(self, inputShape, model_name):
        # 77.13%
        # Mehr Overfitting, minimal weniger Qualität

        resNet50V2 = keras.applications.ResNet50V2(include_top=False, 
                                                   weights="imagenet",
                                                   input_shape=inputShape)

        for layer in resNet50V2.layers:
            layer.trainable = False

        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=inputShape))

        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.1))
        model.add(keras.layers.RandomZoom(0.1))
        model.add(keras.layers.RandomContrast(0.2))

        model.add(keras.layers.Lambda(
            keras.applications.resnet_v2.preprocess_input
        ))
        model.add(resNet50V2)

        model.add(keras.layers.GlobalAveragePooling2D())

        # Nochmal weiteren Dense-Layer hinzugefügt
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dense(units=10, activation='softmax'))

        return ModelSpec(model=model, name=model_name)
    
    def getTransferLearningModelResNet50V2_5(self, inputShape, model_name):
        # 77.27%
        # Interessanterweise noch mehr Overfitting und weiterhin etwas geringere Qualität als in Modell 3

        resNet50V2 = keras.applications.ResNet50V2(include_top=False, 
                                                   weights="imagenet",
                                                   input_shape=inputShape)

        for layer in resNet50V2.layers:
            layer.trainable = False

        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=inputShape))

        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.1))
        model.add(keras.layers.RandomZoom(0.1))
        model.add(keras.layers.RandomContrast(0.2))

        model.add(keras.layers.Lambda(
            keras.applications.resnet_v2.preprocess_input
        ))
        model.add(resNet50V2)

        model.add(keras.layers.GlobalAveragePooling2D())

        # BatchNormalisierung hinzugefügt
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(units=10, activation='softmax'))

        return ModelSpec(model=model, name=model_name)
    
    def getTransferLearningModelResNet50V2_6(self, inputShape, model_name):
        # 77.75%
        # Weniger Overfitting, wieder mehr Qualität

        resNet50V2 = keras.applications.ResNet50V2(include_top=False, 
                                                   weights="imagenet",
                                                   input_shape=inputShape)

        for layer in resNet50V2.layers:
            layer.trainable = False

        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=inputShape))

        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.1))
        model.add(keras.layers.RandomZoom(0.1))
        model.add(keras.layers.RandomContrast(0.2))

        model.add(keras.layers.Lambda(
            keras.applications.resnet_v2.preprocess_input
        ))
        model.add(resNet50V2)

        model.add(keras.layers.GlobalAveragePooling2D())

        # DropOut hinzugefügt
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=10, activation='softmax'))

        return ModelSpec(model=model, name=model_name)
    
    def getTransferLearningModelResNet50V2_7(self, inputShape, model_name):
        # 78.17%
        # Mehr Qualität, nicht mehr Overfitting

        resNet50V2 = keras.applications.ResNet50V2(include_top=False, 
                                                   weights="imagenet",
                                                   input_shape=inputShape)

        for layer in resNet50V2.layers:
            layer.trainable = False

        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=inputShape))

        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.1))
        model.add(keras.layers.RandomZoom(0.1))
        model.add(keras.layers.RandomContrast(0.2))

        model.add(keras.layers.Lambda(
            keras.applications.resnet_v2.preprocess_input
        ))
        model.add(resNet50V2)

        model.add(keras.layers.GlobalAveragePooling2D())

        # Weiteren Dense/Batch/Dropout-Block hinzugefügt
        model.add(keras.layers.Dense(units=512, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=10, activation='softmax'))

        return ModelSpec(model=model, name=model_name)
    
    def getTransferLearningModelResNet50V2_8(self, inputShape, model_name):
        # 77.95%
        # Weniger Qualität, Overfitting ähnlich

        resNet50V2 = keras.applications.ResNet50V2(include_top=False, 
                                                   weights="imagenet",
                                                   input_shape=inputShape)

        for layer in resNet50V2.layers:
            layer.trainable = False

        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=inputShape))

        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.1))
        model.add(keras.layers.RandomZoom(0.1))
        model.add(keras.layers.RandomContrast(0.2))

        model.add(keras.layers.Lambda(
            keras.applications.resnet_v2.preprocess_input
        ))
        model.add(resNet50V2)

        model.add(keras.layers.GlobalAveragePooling2D())

        # Nochmal weiteren Dense/Batch/Dropout-Block hinzugefügt
        model.add(keras.layers.Dense(units=512, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=512, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=10, activation='softmax'))

        return ModelSpec(model=model, name=model_name)
    
    def fineTuneResNet50V2(self, input_model_path, trainable_layers, model_name):
        model = keras.models.load_model(
            input_model_path,
            custom_objects={
                "preprocess_input": keras.applications.resnet_v2.preprocess_input
            }
        )

        base_model = model.get_layer("resnet50v2")
        base_model.trainable = True

        total_layers = len(base_model.layers)
        trainable_layers = max(0, min(trainable_layers, total_layers))

        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

        return ModelSpec(model=model, name=model_name)