import cv2
import numpy as np
from typing import List, Iterable
from mediapipe.python.solutions.face_mesh import FaceMesh


def detect_landmarks(src: np.ndarray, is_stream: bool = False):
    """
    Given an image `src` retrieves the facial landmarks associated with it
    """
    with FaceMesh(static_image_mode=not is_stream, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None


def normalize_landmarks(landmarks, height: int, width: int, mask: Iterable = None):
    normalized_landmarks = np.array([(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks])
    if mask:
        normalized_landmarks = normalized_landmarks[mask]
    return normalized_landmarks


def plot_landmarks(src: np.array, landmarks: List, show: bool = False):
    dst = src.copy()
    for x, y in landmarks:
        cv2.circle(dst, (x, y), 2, 0, cv2.FILLED)
    if show:
        print("Displaying image plotted with landmarks")
        cv2.imshow("Plotted Landmarks", dst)
    return dst
