from io import BytesIO
import numpy as np
import requests
from PIL import Image
from merged_model import CompletedModel

model = None


def load_model():
    print("Model loading.....")
    model = CompletedModel()
    print("Model loaded")

    return model


def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()

    img = np.asarray(image)
    _, _, result = model.predict(img)
    print(result)

    return result


def read_image_file(file) -> Image.Image:
    try:
        response = requests.get(file)
        image = Image.open(BytesIO(response.content))
    except:
        image = Image.open(BytesIO(file))
    return image