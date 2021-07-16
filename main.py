from utils import *

# Video Input from Webcam
video_capture = cv2.VideoCapture(0)
while True:
    ret_val, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    if ret_val:
        cv2.imshow("Original", frame)
        feat_applied = apply_makeup(frame, True, 'lips', False)
        cv2.imshow("Feature", feat_applied)

        if cv2.waitKey(1) == 27:
            break

# # Static Images
# image = cv2.imread("model.jpg", cv2.IMREAD_UNCHANGED)
# output = apply_makeup(image, False, 'foundation', False)
#
# cv2.imshow("Original", image)
# cv2.imshow("Feature", output)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
