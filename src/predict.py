from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "emergency_vehicle_classifier.keras"
)

loaded_model = load_model(MODEL_PATH)

def predict_vehicle(img):

    img = img.resize((256,256))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    probability = loaded_model.predict(img_array, verbose=0)[0][0]

    if probability >= 0.5:
        label = "Emergency Vehicle"
    else:
        label = "Non-Emergency Vehicle"



    return label, probability