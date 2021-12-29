# ======= imports
import numpy as np
import cv2

pass

# ======= constants
pass

# === template image keypoint and descriptors
template = cv2.imread("template.jpg")
template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
template_rgb_height, template_rgb_width = template_rgb.shape[:2]
template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)

feature_extractor = cv2.SIFT_create()
keyPoints_template, description_template = feature_extractor.detectAndCompute(template_gray, None)

pass

ref_image = cv2.imread("refImage.jpg")
ref_image = cv2.resize(ref_image, (template_rgb_width, template_rgb_height))

white_array = np.zeros((template_rgb.shape[0], template_rgb.shape[1], 3), dtype=np.uint8)
white_array.fill(255)
# ===== video input, output and metadata
template_video = cap = cv2.VideoCapture("templateVideo.mp4")
ret, frame = cap.read()

pass
out = cv2.VideoWriter('PerspectiveWarping.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame.shape[1], frame.shape[0]))
# ========== run on all frames
while cap.isOpened() and ret:

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    keyPoints_frame, description_frame = feature_extractor.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(description_template, description_frame, k=2)
    good_and_second_good_match_list = [m for m in matches if m[0].distance / m[1].distance < 0.6]
    good_match_arr = np.asarray(good_and_second_good_match_list)[:, 0]

    pass
    if good_match_arr.shape[0] > 3:

        good_kp_template = np.array([keyPoints_template[m.queryIdx].pt for m in good_match_arr])
        good_kp_frame = np.array([keyPoints_frame[m.trainIdx].pt for m in good_match_arr])
        H, masked = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)
        # ======== find homography
        # also in SIFT notebook

        pass

        # ++++++++ do warping of another image on template image
        # we saw this in SIFT notebook
        wrap_louver_and_frame = cv2.warpPerspective(ref_image, H, (frame.shape[1], frame.shape[0]))
        wrap_white_array_and_frame = cv2.warpPerspective(white_array, H, (frame.shape[1], frame.shape[0]))

        wrap_white_array_and_frame = cv2.bitwise_not(wrap_white_array_and_frame)
        frame_image_with_black_square = cv2.bitwise_and(wrap_white_array_and_frame, frame)

        result = cv2.bitwise_or(frame_image_with_black_square, wrap_louver_and_frame)

    pass

    # =========== plot and save frame
    cv2.imshow('result', result)
    out.write(result)
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    pass

# ======== end all

cap.release()
out.release()
cv2.destroyAllWindows()
pass
