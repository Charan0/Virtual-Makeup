import cv2
from landmarks import detect_landmarks, normalize_landmarks, plot_landmarks
from utils import blush_mask, lip_mask, mask_skin, gamma_correction
import numpy as np

upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 408, 415, 272, 271, 268, 12, 38, 41, 42, 191, 78, 76]
lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
face_conn = [10, 338, 297, 332, 284, 251, 389, 264, 447, 376, 433, 288, 367, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 138, 213, 147, 234, 127, 162, 21, 54, 103, 67, 109]
cheeks = [425, 205]


def apply_makeup(src: np.ndarray, is_stream: bool, feature: str, show_landmarks: bool):
    """
    Takes in a source image and applies effects onto it.
    """
    ret_landmarks = detect_landmarks(src, is_stream)
    height, width, _ = src.shape
    if feature == 'lips':
        feature_landmarks = normalize_landmarks(ret_landmarks, height, width, upper_lip + lower_lip)
        mask = lip_mask(src, feature_landmarks, [153, 0, 157])
        output = cv2.addWeighted(src, 1.0, mask, 0.4, 0.0)
    elif feature == 'blush':  # Defaults to blush for any other thing
        feature_landmarks = normalize_landmarks(ret_landmarks, height, width, cheeks)
        mask = blush_mask(src, feature_landmarks, [153, 0, 157], 50)
        output = cv2.addWeighted(src, 1.0, mask, 0.3, 0.0)
    else:
        skin_mask = mask_skin(src)
        output = np.where(src * skin_mask >= 1, gamma_correction(src, 1.75), src)
    if show_landmarks:
        plot_landmarks(src, feature_landmarks, True)
    return output


# # Video Input from Webcam
# video_capture = cv2.VideoCapture(0)
# while True:
#     ret_val, frame = video_capture.read()
#     frame = cv2.flip(frame, 1)
#     if ret_val:
#         cv2.imshow("Original", frame)
#         feat_applied = apply_makeup(frame, True, 'blush', True)
#         cv2.imshow("Feature", feat_applied)
#
#         if cv2.waitKey(1) == 27:
#             break

# Static Images
image = cv2.imread("model.jpg", cv2.IMREAD_UNCHANGED)
output = apply_makeup(image, False, 'foundation', False)

cv2.imshow("Original", image)
cv2.imshow("Feature", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
