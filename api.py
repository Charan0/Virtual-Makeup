"""
API Endpoints using FastAPI for virtual try-on.
TODO: Synchronous API endpoints are sufficient. Remove all async, await
"""

import numpy as np
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
from utils import apply_makeup, apply_feature
from starlette.responses import StreamingResponse
import cv2
import enum
from typing import List

app = FastAPI(title="API endpoints for virtual makeup",
              description="These API endpoints can be used to try virtual face makeup - lip_color, blush, foundation")


class FeatureChoice(str, enum.Enum):
    """
    An Enum for choice of feature.
    """
    lips = 'lips'
    blush = 'blush'
    foundation = 'foundation'


@app.get('/')
def root():
    return {"title": "Well...\nHello there! ",
            "message": "Nothing much to see here but HEY! try out the other endpoints. "
                       "Hope you like them, you can read more about them at http://127.0.0.1:8000/docs"}


@app.get('/apply-makeup/')
def info_try_makeup():
    """
    ### Information about the post request on the same route.
    """
    return {
        "message": "Perform a post request on the same route",
        "info": "A post request on this route with the necessary query parameters (choice, file) "
                "returns an image with the feature applied."
    }


@app.post('/apply-makeup/')
async def try_makeup(choice: FeatureChoice, file: UploadFile = File(...)):
    """
    Given a choice (`lips`, `blush`, `foundation`) and an input image returns the output with the applied feature
    """
    image = np.array(Image.open(BytesIO(await file.read())))
    output = cv2.cvtColor(apply_makeup(image, False, choice.value, False), cv2.COLOR_BGR2RGB)
    ret_val, output = cv2.imencode(".png", output)
    return StreamingResponse(BytesIO(output), media_type="image/png")


@app.get('/apply-feature/')
def info_try_feature():
    """
    ### Information about the post request on the same route.
    """
    return {
        "message": "Perform a post request on the same route",
        "info": "A post request on this route with the necessary query parameters (choice, file) "
                "returns an image with the feature applied.",
        "Note": "This method is specifically to reduce the processing load on the server, "
                "supply this with normalized landmark coordinates for best performance"
    }


@app.post('/apply-feature/')
async def try_feature(choice: FeatureChoice, landmarks: List[List[int]], normalize: bool,
                      file: UploadFile = File(...)):
    """
    Given a choice (`lips`, `blush`, `foundation`) and an input image returns the output with the applied feature.
    Specifically to **reduce the processing load on the server**, preferably detect and normalize the landmarks
    before making a call to this endpoint
    """
    image = np.array(Image.open(BytesIO(await file.read())))
    output = await apply_feature(image, choice, landmarks, normalize, False)
    return StreamingResponse(BytesIO(output), media_type="image/png")
