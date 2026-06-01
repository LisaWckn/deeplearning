import keras
from dataclasses import dataclass

@dataclass
class ModelSpec:
    model: keras.Model
    name: str

class Models:

    def __init__(self, inputShape=(224,224,3)):
        self.model1 = self.getModel1(inputShape, model_name='model1')
        
        self.model2 = self.getModel2(inputShape, model_name='model2')

        self.model3 = self.getModel3(inputShape, model_name='model3')
        self.model3_withEarlyStopping = self.getModel3(inputShape, model_name='model3_withEarlyStopping')
        self.model3_withEarlyStoppingAndReduceLR = self.getModel3(inputShape, model_name='model3_withEarlyStoppingAndReduceLR')

        self.model4 = self.getModel4(inputShape, model_name='model4')

        self.model5 = self.getModel5(inputShape, model_name='model5')

        self.model6 = self.getModel6(inputShape, model_name='model6')

        self.model7 = self.getModel7(inputShape, model_name='model7')
        
        self.model8 = self.getModel8(inputShape, model_name='model8')
        
        self.model9 = self.getModel9(inputShape, model_name='model9')
        
        self.model10 = self.getModel10(inputShape, model_name='model10')
        
        self.model11 = self.getModel11(inputShape, model_name='model11')
        
        self.model12 = self.getModel12(inputShape, model_name='model12')
        
        self.model13 = self.getModel13(inputShape, model_name='model13')
        
        self.model14 = self.getModel14(inputShape, model_name='model14')
        
        self.model15 = self.getModel15(inputShape, model_name='model15')
        
        self.model16 = self.getModel16(inputShape, model_name='model16')
        
        self.model17 = self.getModel17(inputShape, model_name='model17')
        
        self.model18 = self.getModel18(inputShape, model_name='model18')
        
        self.model19 = self.getModel19(inputShape, model_name='model19')

    def getModel1(self, inputShape, model_name):

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
        model.add(keras.layers.Dense(units=10, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name)
    
    def getModel2(self, inputShape, model_name):

        model = keras.Sequential()

        # Normalization added
        model.add(keras.layers.Rescaling(1./255))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name)
    
    def getModel3(self, inputShape, model_name):

        model = keras.Sequential()

        # Augmenation added
        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name)
    
    def getModel4(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        # Elu statt Relu getestet
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='elu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='elu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='elu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='elu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='elu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='elu'))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name)
    
    def getModel5(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))
        
        # Swish/Silu statt Elu getestet
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='silu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='silu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='silu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='silu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='silu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='silu'))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name)
    
    def getModel6(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        # BatchNormalization Layer
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=inputShape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name)
    
    def getModel7(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        # Dropout Layer
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name)
    
    def getModel8(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        # GlobalAveragePooling statt Flatten
        model.add(keras.layers.GlobalAveragePooling2D())

        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name)
    
    def getModel9(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        # Padding same
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())        
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name)
    
    def getModel10(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        # kernel_initializer='he_normal' statt default 'glorot_uniform
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape, kernel_initializer='he_normal'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())        
        model.add(keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name)
    
    def getModel11(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        # Andere Aufteilung der Conv2D Layer und mehr FeatureMaps im zweiten Block
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 

        return ModelSpec(model=model, name=model_name)
    
    def getModel12(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        # Dritter Block mit noch mehr FeatureMaps
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 

        return ModelSpec(model=model, name=model_name)
    
    def getModel13(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        # Weiteres Dense Layer für bessere Kategorisierung
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 

        return ModelSpec(model=model, name=model_name)
    
    def getModel14(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        # Wieder GlobalAveragePooling
        model.add(keras.layers.GlobalAveragePooling2D())

        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 

        return ModelSpec(model=model, name=model_name)
    
    def getModel15(self, inputShape, model_name):

        model = keras.Sequential()

        # RandomFlip angepasst
        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.GlobalAveragePooling2D())

        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 

        return ModelSpec(model=model, name=model_name)
    
    def getModel16(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        # BatchNormalisierung erneut getestet
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.GlobalAveragePooling2D())

        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 

        return ModelSpec(model=model, name=model_name)
    
    def getModel17(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.GlobalAveragePooling2D())

        # Weitere Dropout Schicht
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 

        return ModelSpec(model=model, name=model_name)
    
    def getModel18(self, inputShape, model_name):

        model = keras.Sequential()

        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.05))
        model.add(keras.layers.RandomZoom(0.05))
        model.add(keras.layers.RandomContrast(0.1))

        model.add(keras.layers.Rescaling(1./255))

        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        # Weitere Conv2d Schicht mit mehr Feature Maps
        model.add(keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))

        model.add(keras.layers.GlobalAveragePooling2D())

        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 

        return ModelSpec(model=model, name=model_name)

    def getModel19(self, inputShape, model_name):

        model = keras.Sequential()
        
        model.add(keras.layers.RandomFlip("horizontal"))
        model.add(keras.layers.RandomRotation(0.1))
        model.add(keras.layers.RandomZoom(0.1))
        model.add(keras.layers.RandomContrast(0.2))

        model.add(keras.layers.Rescaling(1./255))
        
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', input_shape=inputShape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        
        model.add(keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        
        model.add(keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
        
        model.add(keras.layers.GlobalAveragePooling2D())
        
        model.add(keras.layers.Dense(units=512, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(units=256, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(units=10, activation='softmax')) 
        
        return ModelSpec(model=model, name=model_name)