# ======= imports
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
import mesh_renderer


def camera_calibration():
    square_size = 2.88
    img_mask = "../MyChess/*.jpeg"
    pattern_size = (9, 6)
    img_names = glob(img_mask)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    obj_points = []
    img_points = []
    img = cv2.imread(img_names[0])
    h, w = img.shape[:2]
    for i, file_name in enumerate(img_names):
        imgBGR = cv2.imread(file_name)
        imgBGR = cv2.resize(imgBGR, (w, h))
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

        assert w == img.shape[1] and h == img.shape[0], f"size: {img.shape[1]} x {img.shape[0]}"
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if not found:
            print("chessboard not found")
            continue
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

    # plt.show()
    return cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)


pass
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = camera_calibration()
# ======= constants

pixel_to_cm_convert = 0.0264583333

pass

# === template image keypoint and descriptors
template = cv2.imread("template.jpg")
template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
template_rgb_height, template_rgb_width = template_rgb.shape[:2]
template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)

feature_extractor = cv2.SIFT_create()
keyPoints_template, description_template = feature_extractor.detectAndCompute(template_gray, None)

pass

# ===== video input, output and metadata
template_video = cap = cv2.VideoCapture("templateVideo.mp4")
ret, frame = cap.read()
pass
frame_h, frame_w = frame.shape[:2]
mesh = mesh_renderer.MeshRenderer(K=camera_matrix, video_width=frame_w, video_height=frame_h,
                                  obj_path=".\drill\drill.obj")
# ========== run on all frames
out = cv2.VideoWriter('AugmentedReality.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (frame.shape[1], frame.shape[0]))
while cap.isOpened() and ret:
    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    keyPoints_frame, description_frame = feature_extractor.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(description_template, description_frame, k=2)
    good_and_second_good_match_list = [m for m in matches if m[0].distance / m[1].distance < 0.45]
    good_match_arr = np.asarray(good_and_second_good_match_list)[:, 0]
    pass

    # ======== find homography
    # also in SIFT notebook
    if len(good_match_arr) > 3:
        good_kp_template = np.array([keyPoints_template[m.queryIdx].pt for m in good_match_arr])
        good_kp_frame = np.array([keyPoints_frame[m.trainIdx].pt for m in good_match_arr])
        H, masked = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)
        filter_good_kp_frame = np.array([good_kp_frame[i] for i in range(len(masked)) if masked[i] == 1])
        filter_good_kp_template = np.array([good_kp_template[i] for i in range(len(masked)) if masked[i] == 1])
        z = np.zeros((filter_good_kp_template.shape[0], 1))
        filter_good_kp_template *= pixel_to_cm_convert
        filter_good_kp_template = np.append(filter_good_kp_template, z, axis=1)
        valid, r_vec, t_vec = cv2.solvePnP(filter_good_kp_template, filter_good_kp_frame, camera_matrix, None)
        if valid:
            drawn_image = mesh.draw(img=frame, rvec=r_vec, tvec=t_vec)

    cv2.imshow('result', drawn_image)
    out.write(drawn_image)
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# ======== end all
pass
