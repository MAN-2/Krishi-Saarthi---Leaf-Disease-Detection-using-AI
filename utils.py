import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

# class name as per CSV file
class_names = pd.read_csv("class_names.csv")["class_names"].tolist()
NUM_CLASSES = len(class_names)


base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None   
)
base_model.trainable = True

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation="softmax")
])

# Load weights
model.load_weights("model/plant_disease_model.h5") #trained model 


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    img = image.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    if arr.shape[-1] == 4:       
        arr = arr[..., :3]
    return np.expand_dims(arr, 0)  # (1, 224, 224, 3)

def predict_image(image: Image.Image):
    x = preprocess_image(image)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return class_names[idx], float(preds[idx])

def predict_top_k(image: Image.Image, k=3):
    x = preprocess_image(image)
    preds = model.predict(x)[0]
    top_idxs = preds.argsort()[::-1][:k]
    return [(class_names[i], float(preds[i])) for i in top_idxs]
