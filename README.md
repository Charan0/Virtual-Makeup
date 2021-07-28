# Virtual-Makeup
Python, OpenCV based virtual tryon for makeup (lipcolor, blush, foundation prolly eyewear too)

These python scripts add "make up" on to an input. The input is either a static image of a person's face or live webcam feed.
Currently only lipcolor and face blush is supported and the color of defaults to `rgb(157, 0, 153)` but it can be changed.

# How to use

1. Clone this repository
2. Create a virtual environment using `python3 -m venv env` or anyother way of creating virtual envs
3. Install the requirements using `pip install -r requirements.txt`
4. To try the makeup process on the included model.jpg comment out the video capture code and uncomment the static image code and run `python main.py`

# Sample outputs from ths implementation


Original Sample |Blush Applied
:-------------------------:|:-------------------------:
![Original Image](https://user-images.githubusercontent.com/40448838/125641690-4cc137cd-4e20-4e8b-bbc6-0d81f1a50f4a.png)  |  ![Light Pink blush applied](https://user-images.githubusercontent.com/40448838/125641612-e5075a25-7ab0-41d4-b1f6-e1d7e55ccccf.png)

Original Sample |Lip Color applied
:-------------------------:|:-------------------------:
![Original Image](https://user-images.githubusercontent.com/40448838/125641792-46761f24-6418-4004-9381-910f9fbe5ef0.png) | ![Image with Lip color applied](https://user-images.githubusercontent.com/40448838/125641817-c0755878-2358-4e51-92bb-87531a2e04da.png)


Original Sample |Foundation applied
:-------------------------:|:-------------------------:
![Original Image](https://user-images.githubusercontent.com/40448838/125849113-2f9e5147-ec37-4d85-991d-fe3b24240d83.png) | ![image](https://user-images.githubusercontent.com/40448838/125849001-54569ffd-194f-4755-a568-5f7f6a854e04.png)



## How it works
Using mediapipe I detect 468 facial landmarks and and pull out the required landmarks (lips and cheek landmarks) and after that I use simple image processing techniques to achieve the end result

## File structure
`main.py` -> Primary file, reads the input image and applies the makeup  
`api.py` -> Contains the fastapi endpoints  
`landmarks.py` -> Contains all the functionality for the landmarks (detection, normalization etc)  
`sample.py` -> A sample script that uses `python-requests` to demonstrate the endpoints  
`utils.py`  -> Contains utility functions.  

## Fast API
Provides openapi spec by default, run using `uvicorn api:app` and navigate to localhost:8000/docs to view the openapi spec or to localhost:8000/redoc to view the redoc spec. Currently two API Endpoints are supported `apply-makeup` and `apply-feature` (Needs Fix, doesn't work as of now)
