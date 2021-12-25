# ======= imports
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
import mesh_renderer


def warpTwoImages(img1, img2, H):
    dst = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))  # wraped image
    return dst

def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


def camera_calibration():
    square_size = 2.88
    img_mask = "../MyChess/*.jpeg"
    pattern_size = (9, 6)
    figsize = (10, 10)
    img_names = glob(img_mask)
    number_of_images = len(img_names)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size
    obj_points = []
    img_points = []
    h, w = cv2.imread(img_names[0]).shape[:2]
    h2, w2 = cv2.imread("1.jpeg").shape[:2]
    for i, file_name in enumerate(img_names):
        imgBGR = cv2.imread(file_name)
        # imgBGR=cv2.resize(imgBGR,(w2,h2))
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

        assert w == img.shape[1] and h == img.shape[0], f"size: {img.shape[1]} x {img.shape[0]}"
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if not found:
            print("chessboard not found")
            continue
        if i < 12:
            img_w_corners = cv2.drawChessboardCorners(imgRGB, pattern_size, corners, found)
            #plt.subplot(4, 3, i + 1)
            #plt.imshow(img_w_corners)

        print(f"{file_name}... OK")
        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

    #plt.show()
    return cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


#  plt.figure(figsize=figsize)
#  for i, fn in enumerate(img_names):
#
# #     imgBGR = cv2.imread(fn)
#      imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
#
#      dst = cv2.undistort(imgRGB, camera_matrix, dist_coefs)
#
#      if i < 12:
#          plt.subplot(4, 3, i + 1)
#          plt.imshow(dst)
#
#  plt.show()
#  print("Done")
#  objectPoints = (
#          3
#          * square_size
#          * np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
#  )
#  plt.figure(figsize=figsize)
#  for i, fn in enumerate(img_names):
#
#      imgBGR = cv2.imread(fn)
#      imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
#
#      dst = cv2.undistort(imgRGB, camera_matrix, dist_coefs)
#
#      imgpts = cv2.projectPoints(objectPoints, _rvecs[i], _tvecs[i], camera_matrix, dist_coefs)[0]
#      drawn_image = draw(dst, imgpts)
#
#      if i < 12:
#          plt.subplot(4, 3, i + 1)
#          plt.imshow(drawn_image)
#
#  plt.show()

pass
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = camera_calibration()
# ======= constants
square_size = 2.88
objectPoints = (
        3
        * square_size
        * np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
)

cm_convert = 0.0264583333
figsize = (10, 10)
pass

# === template image keypoint and descriptors
template = cv2.imread("template.jpg")
template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
template_rgb_height, template_rgb_width = template_rgb.shape[:2]
pts1 = np.float32([[167, 35], [1421, 51], [151, 883], [1447, 879]])
pts2 = np.float32(
    [[0, 0], [template_rgb_width, 0], [0, template_rgb_height], [template_rgb_width, template_rgb_height]])
mat = cv2.getPerspectiveTransform(pts1, pts2)
template_rgb = cv2.warpPerspective(template_rgb, mat, (template_rgb_width, template_rgb_height))
template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_BGR2GRAY)
feature_extractor = cv2.SIFT_create()
keyPoints_template, description_template = feature_extractor.detectAndCompute(template_gray, None)

pass
louver_image = cv2.imread("output.jpg")
louver_rgb = cv2.cvtColor(louver_image, cv2.COLOR_BGR2RGB)
louver_gray = cv2.cvtColor(louver_rgb, cv2.COLOR_BGR2GRAY)
louver_rgb = cv2.resize(louver_rgb, (template_rgb_width, template_rgb_height))
h1, w1 = louver_rgb.shape[:2]

# ===== video input, output and metadata
template_video = cap = cv2.VideoCapture("Video2.mp4")
ret, frame = cap.read()

white_array = np.zeros((template_rgb.shape[0], template_rgb.shape[1], 3), dtype=np.uint8)
white_array.fill(255)
pass
h, w = frame.shape[:2]
mesh = mesh_renderer.MeshRenderer(K=camera_matrix, video_width=w, video_height=h,
                                  obj_path="C:\BScCS\שנה ג\סמסטר א\ראייה ממוחשבת\פרויקטים\Augmented reality\planar_ar\drill\drill.obj")
# ========== run on all frames
while cap.isOpened() and ret:
    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    keyPoints_frame, description_frame = feature_extractor.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(description_template, description_frame, k=2)
    good_and_second_good_match_list = [m for m in matches if m[0].distance / m[1].distance < 0.5]
    good_match_arr = np.asarray(good_and_second_good_match_list)[:, 0]
    pass
    good_kp_template = np.array([keyPoints_template[m.queryIdx].pt for m in good_match_arr])
    good_kp_frame = np.array([keyPoints_frame[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)

    # ======== find homography
    # also in SIFT notebook
    pass
    keyPoints_ref, description_ref = feature_extractor.detectAndCompute(louver_gray, None)
    good_kp_frame = np.array([keyPoints_frame[i].pt for i in range(len(masked)) if masked[i] == 1])
    good_kp_template = np.array([keyPoints_template[i].pt for i in range(len(masked)) if masked[i] == 1])
    z = np.zeros((good_kp_template.shape[0], 1))
    good_kp_template = np.append(good_kp_template, z, axis=1)
    good_kp_template *= cm_convert
    valid,r_vec,t_vec = cv2.solvePnP(good_kp_template, good_kp_frame, camera_matrix, dist_coefs)

    result = warpTwoImages(louver_rgb, frame_rgb, H)
    result2 = warpTwoImages(white_array, frame_rgb, H)
    result2 = cv2.bitwise_not(result2)
    result3 = cv2.bitwise_and(result2, frame_rgb)
    result3 = cv2.bitwise_or(result3, result)
   # imgpts = cv2.projectPoints(objectPoints, _rvecs[0], _tvecs[0], camera_matrix, dist_coefs)[0]
    #drawn_image = draw(result3, imgpts)
    drawn_image=mesh.draw(img=result3,rvec=_rvecs[0],tvec=_tvecs[0])
    cv2.imshow('result', drawn_image)
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # ++++++++ take subset of keypoints that obey homography (both frame and reference)
    # this is at most 3 lines- 2 of which are really the same
    # HINT: the function from above should give you this almost completely
    pass

    # ++++++++ solve PnP to get cam pose (r_vec and t_vec)
    # `cv2.solvePnP` is a function that receives:
    # - xyz of the template in centimeter in camera world (x,3)
    # - uv coordinates (x,2) of frame that corresponds to the xyz triplets
    # - camera K =camera_matrix
    # - camera dist_coeffs =dist_coefs
    # and outputs the camera pose (r_vec and t_vec) such that the uv is aligned with the xyz.
    #
    # NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
    # because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
    # For this we just need the template width and height in cm.
    #
    # this part is 2 rows
    pass

    # ++++++ draw object with r_vec and t_vec on top of rgb frame
    # We saw how to draw cubes in camera calibration. (copy paste)
    # after this works you can replace this with the draw function from the renderer class renderer.draw() (1 line)
    pass

    # =========== plot and save frame
    pass

# ======== end all
pass
