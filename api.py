import enum

import numpy as np
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from utils import apply_makeup
from starlette.responses import StreamingResponse
import cv2

app = FastAPI(title="API endpoints for virtual makeup",
              description="These API endpoints can be used to try virtual face makeup - lip-color, blush, foundation")


class FeatureChoice(str, enum.Enum):
    """
    An Enum for choice of feature
    """
    lips = 'lips'
    blush = 'blush'
    foundation = 'foundation'


@app.get('/')
def root():
    return {"title": "Welcome",
            "message": "Nothing much to see here but HEY! try out the other endpoints. Hope you like them"}


@app.post('/upload-file/')
async def try_makeup(choice: FeatureChoice, file: UploadFile = File(...)):
    """
    Given a choice (`lips`, `blush`, `foundation`) and an input image returns the output with the applied feature
    """
    image = np.array(Image.open(BytesIO(await file.read())))
    output = cv2.cvtColor(apply_makeup(image, False, choice.value, False), cv2.COLOR_BGR2RGB)
    ret_val, output = cv2.imencode(".png", output)
    return StreamingResponse(BytesIO(output), media_type="image/png")
