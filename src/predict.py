from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

loaded_model = load_model("../model/emergency_vehicle_classifier.keras")

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