import gradio as gr
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image
import os

# Model file name
model_file = "efficientnetv2b2 (1).keras"

# Download from Google Drive if not present
if not os.path.exists(model_file):
    gdown.download(
       "https://drive.google.com/uc?id=10cfH3AeLud37TuQ-abrv67UPLPdFlF1e", 
        model_file, quiet=False
    )

# Load model
model = tf.keras.models.load_model(model_file)

# Class labels
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Prediction function
def predict(img):
    img = img.resize((124, 124))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    return {class_names[i]: float(preds[i]) for i in range(len(class_names))}

# Gradio interface
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Garbage Classification",
    description="Upload an image to classify the type of waste."
)

app.launch()
