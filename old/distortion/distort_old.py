import random
import cv2
import numpy as np


def _distort(image, k=-400e-8, dx=0, dy=0):
    """Distort image with pincushion/barrel distortion.
    Taken from: https://www.programcreek.com/python/example/89322/cv2
    """
    height, width, channel = image.shape
    print(height, width, channel)
    #k = random.uniform(-distort_limit, distort_limit) * 0.00001
    #dx = random.uniform(-shift_limit, shift_limit) * width
    #dy = random.uniform(-shift_limit, shift_limit) * height
    y, x = np.mgrid[0:height:1, 0:width:1]
    x = x.astype(np.float32) - width / 2 - dx
    y = y.astype(np.float32) - height / 2 - dy
    theta = np.arctan2(y, x)
    r = (x * x + y * y) ** 0.5
    rd = r * (1 - k * r * r)
    map_x = rd * np.cos(theta) + width / 2 + dx
    map_y = rd * np.sin(theta) + height / 2 + dy
    image_distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #image_distorted = _crop_border(image_distorted)
    return image_distorted


def _crop_border(image):
    """Crops image so no black border occurs after remap."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    image_cropped = image[y:y+h, x:x+w]
    return image_cropped


def _undistort(image, k=-400e-8, dx=0, dy=0, scaling_factor=1.0):
    """Undistort a given image based on the known distortion factor."""
    height, width, channel = image.shape
    y, x = np.mgrid[0:height:1, 0:width:1]
    x = x.astype(np.float32) - width / 2 - dx
    y = y.astype(np.float32) - height / 2 - dy
    theta = np.arctan2(y, x)
    r = (x * x + y * y) ** 0.5
    rd = r * (1 + k * r * r)
    map_x = scaling_factor * rd * np.cos(theta) + width / 2 + dx
    map_y = scaling_factor * rd * np.sin(theta) + height / 2 + dy
    image_undistorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return image_undistorted


# def _undistort_cv(image, k):
    # """Undistort a given image based on the known distortion factor."""
    # height, width, channel = image.shape
    # K = np.eye(3)
    # f = 200
    # K[0, 0] = f
    # K[1, 1] = f
    # K[0, 2] = np.shape(image)[1]/2
    # K[1, 2] = np.shape(image)[0]/2
    # print(K)
    # R = np.eye(3)
    # P = np.eye(3)
    # D = np.array([k*1e8*0.001, 0., 0., 0.])
    # K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), np.eye(3), balance=1.0, fov_scale=2.0)
    # print(K_new)
    # map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K, D, R, K_new, (width, height), cv2.CV_16SC2)
    # image_undistorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # return image_undistorted


img_dir = "imagenet/img.jpg"
image = cv2.imread(img_dir)

k = -600e-8

image_distorted = _distort(image, k, dx=0, dy=0)
image_undistorted = _undistort(image, k, dx=0, dy=0, scaling_factor=4.0)  # for normal image 2.1

while True:
    cv2.imshow("img_original", image)
    cv2.imshow("img_distorted", image_distorted)
    cv2.imshow("image_undistorted", image_undistorted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
