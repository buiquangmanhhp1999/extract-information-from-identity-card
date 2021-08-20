import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from serve_model import *
from pydantic import BaseModel

app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>"""

app = FastAPI(title="Tensorflow FastAPI Start Pack", description=app_desc)


class Item(BaseModel):
    file: str


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image/url")
async def predict_api(file: Item):
    try:
        extension = file.file.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            return "Image must be jpg or png format!"

        image = read_image_file(file.file)
        prediction = predict(image)

        return prediction
    except Exception as e:
        print("Payload Exception" + str(e))


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    try:
        extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            return "Image must be jpg or png format!"

        image = read_image_file(await file.read())
        prediction = predict(image)

        return prediction
    except Exception as e:
        print("Payload File Exception" + str(e))


if __name__ == "__main__":
    uvicorn.run(app, debug=True)