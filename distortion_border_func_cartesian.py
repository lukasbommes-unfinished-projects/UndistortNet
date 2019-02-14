import random
import cv2
import numpy as np

import matplotlib.pyplot as plt

# TODO:
# - write function to map points from distorted to undistorted image
#   (undistorted points can then be used to find the ground plane homography)
# - crop distorted image so no background is visible
# - scale undistorted image so it fits into the viewport
# (see lensdistort.m in downloads)


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

    # dx=50
    # dy=50
    # k=-0.4
    #
    # image = cv2.imread("old/dataset_experiments/img.jpg")
    # draw_central_rectangle(image, dw=130, dh=130, dx=0, dy=0, color=(0, 0, 255))
    #
    # # crop out largest possible square
    # image = square_center_crop(image)
    # height, width, channel = image.shape
    #
    # # compute distortion and undistortion maps
    # d_map_x, d_map_y = compute_distort_maps(width, height, k, dx, dy)
    # ud_map_x, ud_map_y = compute_undistort_maps(width, height, k, dx, dy)
    #
    # # distort image
    # image_distorted = cv2.remap(image, d_map_x, d_map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    #
    # # crop out a region of image_distorted
    # dw=265
    # dh=dw
    # xmin = int(width/2-dw+dx)
    # ymin = int(height/2-dh+dy)
    # xmax = int(width/2+dw+dx+1)
    # ymax = int(height/2+dh+dy+1)
    # image_distorted_cropped = image_distorted[ymin:ymax, xmin:xmax, :]
    #
    # draw_central_rectangle(image_distorted, dw=230, dh=dw, dx=dx/2, dy=dx/2, color=(0, 255, 0))
    #
    # # # resize distorted and cropped image to original size
    # image_distorted_cropped_resized = cv2.resize(image_distorted_cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)
    #
    # # undistort the distorted image
    # image_undistorted = cv2.remap(image_distorted_cropped_resized, ud_map_x, ud_map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    #
    #
    # while True:
    #     cv2.imshow("img_original", image)
    #     cv2.imshow("img_distorted", image_distorted)
    #     cv2.imshow("image_distorted_cropped", image_distorted_cropped)
    #     #cv2.imshow("image_distorted_cropped_resized", image_distorted_cropped_resized)
    #     cv2.imshow("image_undistorted", image_undistorted)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # cv2.destroyAllWindows()


    ##############################################################
    # distortion

    k = -0.4  # for normalized coordinates
    dx = 0
    dy = 0

    h = 30
    w = 50

    yu, xu = np.mgrid[0:h, 0:w]

    # yu:             xu:
    # 0 1 2 .. h-1    0   0   0   .. 0
    # 0 1 2 .. h-1    1   1   1   .. 1
    # . . .     .     .   .   .      .
    # 0 1 2 .. h-1    w-1 w-1 w-1 .. w-1
    #
    #

    # normalize coordinates to range [-0.5..0.5 x -0.5..0.5]
    xur = (xu - dx)/w - 1/2
    yur = (yu - dy)/h - 1/2

    # convert to polar
    ru = np.sqrt(xur * xur + yur * yur)
    theta = np.arctan2(yur, xur)

    # distort coordinates
    rd = ru / (1 - k * ru * ru)

    # def border_correct(xur, yur, xdr, ydr, mode):
    #     assert mode == 'fit' or mode == 'crop', "Invalid mode. Can be either 'fit' or 'crop'."
    #     if mode == "fit":
    #         center_x = round(w/2+dx)
    #         center_y = round(h/2+dy)
    #         #pt_left_center = (xdr[center_y, 0], ydr[center_y, 0])
    #         #pt_right_center = (xdr[center_y, -1], ydr[center_y, -1])
    #         #pt_center_top = (xdr[-1, center_x], ydr[-1, center_x])
    #         #pt_center_bottom = (xdr[0, center_x], ydr[0, center_x])
    #
    #         # get maximum width and height of distorted image
    #         xdr_min, xdr_max = xdr[center_y, 0], xdr[center_y, -1]
    #         ydr_min, ydr_max = ydr[0, center_x], ydr[-1, center_x]
    #         width_d = xdr_max - xdr_min
    #         height_d = ydr_max - ydr_min
    #
    #         # get maximum width and height of undistorted image
    #         xur_min, xur_max = xur[center_y, 0], xur[center_y, -1]
    #         yur_min, yur_max = yur[0, center_x], yur[-1, center_x]
    #         width_u = xur_max - xur_min
    #         height_u = yur_max - yur_min
    #
    #         # get scaling factors
    #         sx = np.abs(xur_max - xur_min) / np.abs(xdr_max - xdr_min)
    #         sy = np.abs(yur_max - yur_min) / np.abs(ydr_max - ydr_min)
    #
    #         # scale meshgrid in x and y direction
    #         xdr = xdr * sx
    #         ydr = ydr * sy
    #
    #         # remove offset
    #         x_offset = xur[center_y, 0] - xdr[center_y, 0]
    #         y_offset = yur[0, center_x] - ydr[0, center_x]
    #         xdr = xdr + x_offset
    #         ydr = ydr + y_offset
    #
    #     elif mode == "crop":
    #         # get minimum width and height of distorted image
    #         xdr_min = np.max((xdr[0, 0], xdr[-1, 0]))
    #         xdr_max = np.min((xdr[0, -1], xdr[-1, -1]))
    #         ydr_min = np.max((ydr[0, 0], ydr[-1, 0]))
    #         ydr_max = np.min((ydr[0, -1], ydr[-1, -1]))
    #         width_d = xdr_max - xdr_min
    #         height_d = ydr_max - ydr_min
    #
    #         print(xdr_min, xdr_max)
    #         print(ydr_min, ydr_max)
    #
    #         # get minimum width and height of undistorted image
    #         xur_min = np.max((xur[0, 0], xur[-1, 0]))
    #         xur_max = np.min((xur[0, -1], xur[-1, -1]))
    #         yur_min = np.max((yur[0, 0], yur[-1, 0]))
    #         yur_max = np.min((yur[0, -1], yur[-1, -1]))
    #         width_u = xur_max - xur_min
    #         height_u = yur_max - yur_min
    #
    #         # get scaling factors
    #         sx = np.abs(xur_max - xur_min) / np.abs(xdr_max - xdr_min)
    #         sy = np.abs(yur_max - yur_min) / np.abs(ydr_max - ydr_min)
    #
    #         # scale meshgrid in x and y direction
    #         xdr = xdr * sx
    #         ydr = ydr * sy
    #
    #         # remove offset
    #         #x_offset = xur[center_y, 0] - xdr[center_y, 0]
    #         #y_offset = yur[0, center_x] - ydr[0, center_x]
    #         #xdr = xdr + x_offset
    #         #ydr = ydr + y_offset
    #
    #     return xdr, ydr

    def border_correct_crop(ru, rd, dx, dy):
        s = ru[0, 0] / rd[0, 0]
        return s

    # rescale (crop or fit)
    s = border_correct_crop(ru, rd, dx, dy)
    rd = rd * s

    # convert back to cartesian
    xdr = rd * np.cos(theta)
    ydr = rd * np.sin(theta)

    # un-normalize coordinates to oriaginal range [0..w x 0...h]
    xd = (xdr + 1/2)*w + dx
    yd = (ydr + 1/2)*h + dy

    f, ax = plt.subplots(1, 1)
    ax.grid()
    ax.scatter(xdr, ydr)
    ax.scatter(xur, yur)
    ax.legend(['undistorted', 'distorted'])
    ax.set_title("distorting points")

    ##############################################################
    # undistortion

    #k = -0.4  # for normalized coordinates
    #dx = 5
    #dy = -5

    #h = 30
    #w = 50

    yd, xd = np.mgrid[0:h, 0:w]

    # normalize coordinates to range [-0.5..0.5 x -0.5..0.5]
    xdr = (xd - dx)/w - 1/2
    ydr = (yd - dy)/h - 1/2

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

    f, ax = plt.subplots(1, 1)
    ax.grid()
    ax.scatter(xdr, ydr)
    ax.scatter(xur, yur)
    ax.legend(['distorted', 'undistorted'])
    ax.set_title("undistorting points")
    plt.show()
