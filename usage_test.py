import keras
import numpy as np

MODEL_NAME = "modelFineTuned_Conv5Block2u3"
IMAGE_NAME = "using_laptop_test"

model_path = f"models/{MODEL_NAME}.keras"
image_path = f"new_pictures/{IMAGE_NAME}.jpeg"

model = keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": keras.applications.resnet_v2.preprocess_input
            }
        )

class_names = ['1', '10', '11', '12', '2', '3', '5', '6', '7', '8']

label_map = {
    "1": "using laptop",
    "2": "hugging",
    "3": "sleeping",
    "5": "clapping",
    "6": "dancing",
    "7": "cycling",
    "8": "calling",
    "10": "eating",
    "11": "fighting",
    "12": "listening to music"
}

img = keras.utils.load_img(
    image_path,
    target_size=(224, 224)
)
img_array = keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

pred = np.array(prediction)

pred_index = np.argmax(pred[0])                 
pred_label = class_names[pred_index]            
pred_class = label_map[pred_label]             
pred_prob = pred[0][pred_index]                 

print(f"Vorhersage: {pred_class}")
print(f"Wahrscheinlichkeit: {pred_prob:.2%}")