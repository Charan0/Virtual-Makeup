import cv2
import numpy as np
from mediapipe.python.solutions.face_detection import FaceDetection


def lip_mask(src: np.ndarray, points: np.ndarray, color: list):
    """
    Given a src image, points of lips and a desired color
    Returns a colored mask that can be added to the src
    """
    mask = np.zeros_like(src)  # Create a mask
    mask = cv2.fillPoly(mask, [points], color)  # Mask for the required facial feature
    # Blurring the region, so it looks natural
    # TODO: Get glossy finishes for lip colors, instead of blending in replace the region
    mask = cv2.GaussianBlur(mask, (7, 7), 5)
    return mask


def blush_mask(src: np.ndarray, points: np.ndarray, color: list, radius: int):
    """
    Given a src image, points of the cheeks, desired color and radius
    Returns a colored mask that can be added to the src
    """
    # TODO: Make the effect more subtle
    mask = np.zeros_like(src)  # Mask that will be used for the cheeks
    for point in points:
        mask = cv2.circle(mask, point, radius, color, cv2.FILLED)  # Blush => Color filled circle
        x, y = point[0] - radius, point[1] - radius  # Get the top-left of the mask
        mask[y:y + 2 * radius, x:x + 2 * radius] = vignette(mask[y:y + 2 * radius, x:x + 2 * radius],
                                                            10)  # Vignette on the mask

    return mask


def mask_skin(src: np.ndarray):
    """
    Given a source image of a person (face image)
    returns a mask that can be identified as the skin
    """
    lower = np.array([0, 133, 77], dtype='uint8')  # The lower bound of skin color
    upper = np.array([255, 173, 127], dtype='uint8')  # Upper bound of skin color
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)  # Convert to YCR_CB
    skin_mask = cv2.inRange(dst, lower, upper)  # Get the skin
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)[..., np.newaxis]  # Dilate to fill in blobs

    if skin_mask.ndim != 3:
        skin_mask = np.expand_dims(skin_mask, axis=-1)
    return (skin_mask / 255).astype("uint8")  # A binary mask containing only 1s and 0s


def face_mask(src: np.ndarray, points: np.ndarray):
    """
    Given a list of face landmarks, return a closed polygon mask for the same
    """
    mask = np.zeros_like(src)
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))
    return mask


def clicked_at(event, x, y, flags, params):
    """
    A useful callback that spits out the landmark index when clicked on a particular landmark
    Note: Very sensitive to location, should be clicked exactly on the pixel
    """
    # TODO: Add some atol to np.allclose
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at {x, y}")
        point = np.array([x, y])
        landmarks = params.get("landmarks", None)
        image = params.get("image", None)
        if landmarks is not None and image is not None:
            for idx, landmark in enumerate(landmarks):
                if np.allclose(landmark, point):
                    print(f"Landmark: {idx}")
                    break
            print("Found no landmark close to the click")


def vignette(src: np.ndarray, sigma: int):
    """
    Given a src image and a sigma, returns a vignette of the src
    """
    height, width, _ = src.shape
    kernel_x = cv2.getGaussianKernel(width, sigma)
    kernel_y = cv2.getGaussianKernel(height, sigma)

    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    blurred = cv2.convertScaleAbs(src.copy() * np.expand_dims(mask, axis=-1))
    return blurred


def face_bbox(src: np.ndarray, offset_x: int = 0, offset_y: int = 0):
    """
    Performs face detection on a src image, return bounding box coordinates with
    an optional offset applied to the coordinates
    """
    height, width, _ = src.shape
    with FaceDetection(model_selection=0) as detector:  # 0 -> dist <= 2mts from the camera
        results = detector.process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None
    results = results.detections[0].location_data
    x_min, y_min = results.relative_bounding_box.xmin, results.relative_bounding_box.ymin
    box_height, box_width = results.relative_bounding_box.height, results.relative_bounding_box.width
    x_min = int(width * x_min) - offset_x
    y_min = int(height * y_min) - offset_y
    box_height, box_width = int(height * box_height) + offset_y, int(width * box_width) + offset_x
    return (x_min, y_min), (box_height, box_width)


def gamma_correction(src: np.ndarray, gamma: float, coefficient: int = 1):
    """
    Performs gamma correction on a source image
    gamma > 1 => Darker Image
    gamma < 1 => Brighted Image
    """
    dst = src.copy()
    dst = dst / 255.  # Converted to float64
    dst = coefficient * np.power(dst, gamma)
    dst = (dst * 255).astype('uint8')
    return dst
