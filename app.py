import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = 96

model = load_model("plant_disease_model.h5")

import tensorflow_datasets as tfds
_, info = tfds.load("plant_village", with_info=True)
class_names = info.features["label"].names

def predict(image):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)[0]
    idx = np.argmax(preds)

    return f"Class {idx} ({preds[idx]*100:.2f}%)"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="🌿 Plant Disease Detection",
    description="Upload a leaf image to detect disease"
)

interface.launch()
