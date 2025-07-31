import os
import requests
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Google Drive download link
MODEL_URL = "https://drive.google.com/uc?id=10cfH3AeLud37TuQ-abrv67UPLPdFlF1e"
MODEL_PATH = "efficientnetv2b2 (1).keras"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Download complete!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Update with your actual class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict(image):
    image = image.resize((124, 124))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)[0]
    return {class_names[i]: float(pred[i]) for i in range(len(class_names))}

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Garbage Classifier",
    description="Upload an image of garbage to classify its type."
)

interface.launch(server_name="0.0.0.0", server_port=8080)
