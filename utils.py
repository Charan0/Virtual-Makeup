import cv2
import numpy as np


def lip_mask(src: np.ndarray, points: np.ndarray, color: list):
    """
    Given a src image, points of lips and a desired color
    Returns a colored mask that can be added to the src
    """
    mask = np.zeros_like(src)  # Create a mask
    mask = cv2.fillPoly(mask, [points], (255, 255, 255))  # Mask for the required facial feature
    colored = np.zeros_like(mask)
    colored[:] = color  # Fill color in the mask
    # Contains only the feature in the desired color
    # The final mask that will be added onto the image
    mask = cv2.GaussianBlur(cv2.bitwise_and(colored, mask), (7, 7), 5)
    return mask


def blush_mask(src: np.ndarray, points: np.ndarray, color: list, radius: int):
    """
    Given a src image, points of the cheeks, desired color and radius
    Returns a colored mask that can be added to the src
    """
    mask = np.zeros_like(src)  # Mask that will be used for the cheeks
    for point in points:
        mask = cv2.circle(mask, point, radius, color, cv2.FILLED)  # Blush => Color filled circle
        x, y = point[0] - radius, point[1] - radius  # Get the top-left of the mask
        mask[y:y+2*radius, x:x+2*radius] = vignette(mask[y:y+2*radius, x:x+2*radius], 10)  # Vignette on the mask

    return mask


def clicked_at(event, x, y, flags, params):
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
