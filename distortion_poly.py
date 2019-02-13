import random
import cv2
import numpy as np

import matplotlib.pyplot as plt

# TODO:
# - write function to map points from distorted to undistorted image
#   (undistorted points can then be used to find the ground plane homography)


def compute_distort_maps(image_width, image_height, k=-0.4, dx=0, dy=0):
    """Compute distortion maps.

    Compute map_x and map_y for barrel distortion with specified distortion
    parameters. Maps can be used with cv2.remap.

    Args:
        image_width (int): Image width in pixels of the image for which the
            distort map is computed.

        image_height (int): Image height in pixels.

        k (float): The distortion coefficient. Has to be smaller or equal
            to zero. Typical values lie in the range [-0.4 ... 0].

        dx (float): Offset of the distortion center in x-direction.
        dy (float): Offset of the distortion center in y-direction.

    Returns:
        map_x (numpy.ndarray): Map which yields the undistorted x coordinate as
            a function of the distorted coordinates: xu = map_x(xd, yd).
        map_y (numpy.ndarray): Map which yields the undistorted y coordinate as
            a function of the distorted coordinates: yu = map_y(xd, yd).

    To distort an image with the computed maps, use
    `cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)`
    """
    assert k <= 0, "Distortion parameter k has to be zero or negative."
    w, h = image_width, image_height
    yd, xd = np.mgrid[0:h, 0:w]
    if k < 0:
        # normalize coordinates to range [-0.5..0.5 x -0.5..0.5]
        xdr = xd/w - dx/w - 1/2
        ydr = yd/h - dy/h - 1/2
        # convert to polar
        rd = np.sqrt(xdr * xdr + ydr * ydr)
        theta = np.arctan2(ydr, xdr)
        # distort coordinates
        ru = -1/(2*k*rd)-np.sqrt(1/(4*k*k*rd*rd)+1/k)
        # convert back to cartesian
        xur = ru * np.cos(theta)
        yur = ru * np.sin(theta)
        # un-normalize coordinates to oriaginal range [0..w x 0...h]
        xu = (xur + 1/2)*w + dx
        yu = (yur + 1/2)*h + dy
        map_x = xu.astype(np.float32)
        map_y = yu.astype(np.float32)
    else:
        map_x = xd.astype(np.float32)
        map_y = yd.astype(np.float32)
    return map_x, map_y


def compute_undistort_maps(image_width, image_height, k=-0.4, dx=0, dy=0):
    """Compute undistortion maps.

    Inverse of `compute_distort_maps`. Computes undistortion maps for specified
    distortion parameters.

    Arguments and return values are equivalent to `compute_distort_maps`. However,
    the maps can be used to undistort an image intead of distorting it. Thus,
    map_x yields xd = map_x(xu, yu) and map_y yields yd = map_y(xu, yu). Use the
    computed maps with cv2.remap as explained for `compute_distort_maps`.

    The computed maps have an inverse meaning of the maps computed with
    `compute_distort_maps` when the same set of parameters (k, dx, dy) is used.
    """
    assert k <= 0, "Distortion parameter k has to be zero or negative."
    w, h = image_width, image_height
    yu, xu = np.mgrid[0:h, 0:w]
    if k < 0:
        # normalize coordinates to range [-0.5..0.5 x -0.5..0.5]
        xur = (xu - dx)/w - 1/2
        yur = (yu - dy)/h - 1/2
        # convert to polar
        ru = np.sqrt(xur * xur + yur * yur)
        theta = np.arctan2(yur, xur)
        # distort coordinates
        rd = ru / (1 - k * ru * ru)
        # convert back to cartesian
        xdr = rd * np.cos(theta)
        ydr = rd * np.sin(theta)
        # un-normalize coordinates to oriaginal range [0..w x 0...h]
        xd = (xdr + 1/2)*w + dx
        yd = (ydr + 1/2)*h + dy
        map_x = xd.astype(np.float32)
        map_y = yd.astype(np.float32)
    else:
        map_x = xu.astype(np.float32)
        map_y = yu.astype(np.float32)
    return map_x, map_y


def draw_central_rectangle(image, dw, dh, dx, dy, color=(255, 255, 255)):
    height, width, channel = image.shape
    xmin = int(np.floor(width/2-dw+dx))
    ymin = int(np.floor(height/2-dh+dy))
    xmax = int(np.ceil(width/2+dw+dx))
    ymax = int(np.ceil(height/2+dh+dy))
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color)
    cv2.circle(image, center=(int(width/2+dx), int(height/2+dy)), radius=10, color=color)


def square_center_crop(image, size=None):
    """Crops the image to a square. The shorter edge of the image determines the size after cropping."""
    height, width, channel = image.shape
    if not size:
        size = np.min((height/2, width/2))
    xmin = int(np.floor(width/2-size))
    ymin = int(np.floor(height/2-size))
    xmax = int(np.ceil(width/2+size-1))
    ymax = int(np.ceil(height/2+size-1))
    return image[ymin:ymax, xmin:xmax, :]


if __name__ == "__main__":

    image = cv2.imread("old/dataset_experiments/img.jpg")

    # crop out largest possible square
    image = square_center_crop(image)
    height, width, channel = image.shape

    # compute distortion and undistortion maps
    d_map_x, d_map_y = compute_distort_maps(width, height, k=-0.4, dx=0, dy=0)
    ud_map_x, ud_map_y = compute_undistort_maps(width, height, k=-0.4, dx=0, dy=0)

    # distort image
    image_distorted = cv2.remap(image, d_map_x, d_map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

    # undistort the distorted image
    image_undistorted = cv2.remap(image_distorted, ud_map_x, ud_map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)


    # draw_central_rectangle(image, dw=130, dh=130, dx=dx, dy=dy, color=(0, 0, 255))
    # image_distorted = distort(image, k=-500e-8, dx=dx, dy=dy)
    #
    # # crop out a region of image_distorted
    # height, width, channel = image_distorted.shape
    # dw=196
    # dh=dw
    # xmin = int(np.floor(width/2-dw+dx))
    # ymin = int(np.floor(height/2-dh+dy))
    # xmax = int(np.ceil(width/2+dw+dx)+1)
    # ymax = int(np.ceil(height/2+dh+dy)+1)
    # image_distorted_cropped = image_distorted[ymin:ymax, xmin:xmax, :]
    # draw_central_rectangle(image_distorted, dw=dw, dh=dh, dx=dx, dy=dy, color=(0, 255, 0))
    #
    # # resize distorted and cropped image to original size
    # image_distorted_cropped_resized = cv2.resize(image_distorted_cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)
    #
    # image_undistorted = undistort(image_distorted_cropped_resized, k=-500e-8, dx=dx, dy=dy)

    while True:
        cv2.imshow("img_original", image)
        cv2.imshow("img_distorted", image_distorted)
        cv2.imshow("image_undistorted", image_undistorted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    #
    # ##############################################################
    # # distortion
    #
    # k = -0.4  # for normalized coordinates
    # dx = 0
    # dy = 0
    #
    # h = 30
    # w = 50
    #
    # yu, xu = np.mgrid[0:h, 0:w]
    #
    # # normalize coordinates to range [-0.5..0.5 x -0.5..0.5]
    # xur = (xu - dx)/w - 1/2
    # yur = (yu - dy)/h - 1/2
    #
    # # convert to polar
    # ru = np.sqrt(xur * xur + yur * yur)
    # theta = np.arctan2(yur, xur)
    #
    # # distort coordinates
    # rd = ru / (1 - k * ru * ru)
    #
    # #print("ru", ru[0,0], "rd", rd[0,0])
    #
    # # convert back to cartesian
    # xdr = rd * np.cos(theta)
    # ydr = rd * np.sin(theta)
    #
    # # un-normalize coordinates to oriaginal range [0..w x 0...h]
    # xd = (xdr + 1/2)*w + dx
    # yd = (ydr + 1/2)*h + dy
    #
    # f, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax.grid()
    # ax.scatter(xu, yu)
    # ax.scatter(xd, yd)
    # ax.legend(['undistorted', 'distorted'])
    # ax.set_title("distorting points")
    #
    # ##############################################################
    # # undistortion
    #
    # #xd, yd = np.mgrid[0:h, 0:w]
    # #xd, yd =
    #
    # # normalize coordinates to range [-0.5..0.5 x -0.5..0.5]
    # xdr = xd/w - dx/w - 1/2
    # ydr = yd/h - dy/h - 1/2
    #
    # # convert to polar
    # rd = np.sqrt(xdr * xdr + ydr * ydr)
    # theta = np.arctan2(ydr, xdr)
    #
    # # distort coordinates
    # ru = -1/(2*k*rd)-np.sqrt(1/(4*k*k*rd*rd)+1/k)
    # #print("ru", ru[0,0], "rd", rd[0,0])
    #
    # # convert back to cartesian
    # xur = ru * np.cos(theta)
    # yur = ru * np.sin(theta)
    #
    # # un-normalize coordinates to oriaginal range [0..w x 0...h]
    # xu = (xur + 1/2)*w + dx
    # yu = (yur + 1/2)*h + dy
    #
    # f, ax = plt.subplots(1, 1, figsize=(7, 7))
    # ax.grid()
    # ax.scatter(xd, yd)
    # ax.scatter(xu, yu)
    # ax.legend(['distorted', 'undistorted'])
    # ax.set_title("undistorting points")
    #plt.show()
