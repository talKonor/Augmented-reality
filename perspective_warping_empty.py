# ======= imports
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import cv2


def warpTwoImages(img1, img2, H):
    dst = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))  # wraped image
    return dst


pass

# ======= constants
figsize = (10, 10)
pass

# === template image keypoint and descriptors
template = cv2.imread("template.jpg")
template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
template_rgb_height, template_rgb_width = template_rgb.shape[:2]
pts1 = np.float32([[167, 35], [1421, 51], [151, 883], [1447, 879]])
pts2 = np.float32([[0, 0], [template_rgb_width, 0], [0, template_rgb_height], [template_rgb_width, template_rgb_height]])
mat = cv2.getPerspectiveTransform(pts1, pts2)
template_rgb = cv2.warpPerspective(template_rgb, mat, (template_rgb_width, template_rgb_height))
template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
feature_extractor = cv2.SIFT_create()
keyPoints_template, description_template = feature_extractor.detectAndCompute(template_gray, None)

pass
luver_image = cv2.imread("output2.jpg")
louver_rgb = cv2.cvtColor(luver_image, cv2.COLOR_BGR2RGB)
louver_rgb=cv2.resize(louver_rgb,(template_rgb_width,template_rgb_height))
h1, w1 = louver_rgb.shape[:2]
#pts1 = np.float32([[165, 1], [1401, 16], [150, 882], [1419, 883]])
#pts2 = np.float32([[0, 0], [w1, 0], [0, h1], [w1, h1]])
#mat = cv2.getPerspectiveTransform(pts1, pts2)
#louver_rgb = cv2.warpPerspective(louver_rgb, mat, (w1, h1))
# ===== video input, output and metadata
template_video = cap = cv2.VideoCapture("Video2.mp4")
ret, frame = cap.read()

white_array = np.zeros((template_rgb.shape[0], template_rgb.shape[1], 3), dtype=np.uint8)
white_array.fill(255)
pass

# ========== run on all frames
while cap.isOpened() and ret:
    frame = cv2.resize(frame, (frame.shape[1],frame.shape[0]))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    keyPoints_frame, description_frame = feature_extractor.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(description_template, description_frame, k=2)
    #good_and_second_good_match_list = [m for m in matches if m[0].distance / m[1].distance < 0.5]
    good_match_arr = np.asarray(matches)[:, 0]
    pass
    good_kp_template = np.array([keyPoints_template[m.queryIdx].pt for m in good_match_arr])
    good_kp_frame = np.array([keyPoints_frame[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)
    # ======== find homography
    # also in SIFT notebook

    pass

    result = warpTwoImages(louver_rgb, frame_rgb, H)
    result2 = warpTwoImages(white_array, frame_rgb, H)
    result2 = cv2.bitwise_not(result2)
    result3 = cv2.bitwise_and(result2, frame_rgb)
    result3 = cv2.bitwise_or(result3, result)

    cv2.imshow('result', result)

    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # ++++++++ do warping of another image on template image
    # we saw this in SIFT notebook
    pass

    # =========== plot and save frame
    pass

# ======== end all
cap.release()
cv2.destroyAllWindows()
pass
