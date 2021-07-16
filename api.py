import numpy as np
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from main import apply_makeup
from enum import Enum
from starlette.responses import StreamingResponse
import cv2

app = FastAPI(title="API endpoints for virtual makeup",
              description="These API endpoints can be used to try virtual face makeup - lip-color, blush, foundation")


class FeatureChoice(str, Enum):
    lips = 'lips'
    blush = 'blush'
    foundation = 'foundation'


@app.get('/')
def root():
    return {"title": "Welcome",
            "message": "Nothing much to see here but HEY! try out the other endpoints. Hope you like them"}


@app.post('/upload-file/')
async def upload_file(choice: FeatureChoice, file: UploadFile = File(...)):
    image = np.array(Image.open(BytesIO(await file.read())))
    output = cv2.cvtColor(apply_makeup(image, False, choice.value, False), cv2.COLOR_BGR2RGB)
    ret_val, output = cv2.imencode(".png", output)
    return StreamingResponse(BytesIO(output), media_type="image/png")
