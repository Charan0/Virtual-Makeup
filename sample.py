import cv2
import requests
import numpy as np
from io import BytesIO
from PIL import Image

video_capture = cv2.VideoCapture(0)  # Start video capture from the webcam
while True:
    ret_val, frame = video_capture.read()  # Read each frame
    if ret_val:  # If frame is found
        frame = cv2.flip(frame, 1)  # Flip the frame -> Selfie
        frame_bytes = cv2.imencode(".png", frame)[1].tobytes()  # Convert image data to bytes, to send to the endpoint
        files = {"file": frame_bytes}
        payload = {"choice": "lips"}
        response = requests.post("http://127.0.0.1:8000/upload-file/", files=files, params=payload)  # API Call
        if response.status_code == 200:
            image_content = response.content  # Response from the API
            rec_image = np.array(Image.open(BytesIO(image_content)))  # Convert the bytes response to numpy array
            rec_image = cv2.cvtColor(rec_image, cv2.COLOR_BGR2RGB)
            # Display both the original and the response
            cv2.imshow("Original", frame)
            cv2.imshow("Feature", rec_image)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
