from io import BytesIO
import numpy as np
from PIL import Image
from merged_model import CompletedModel

model = None


def load_model():
    print("Model loading.....")
    model = CompletedModel()
    print("!!! Completed")

    return model


def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()

    img = np.asarray(image)
    result = model.predict(img)

    return result


def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file))

    return image