import random
import cv2
import numpy as np


def distort(image, k=-400e-8, dx=0, dy=0):
    """Distort image with barrel distortion and distortion center (cx, cy).
    For every (xd, yd) in the distorted image the coordinates (xu, yu)
    in the original image are computed. For each (xd, yd) the remap function
    interpolates the pixel value from the neigbouring pixels in the
    original image. Thus, the following maps which yield (xu, yu) for a given
    (xd, yd) are created:
     -> xu = map_x(xd, yd)
     -> yu = map_y(xd, yd)
    The equations implemented to create the maps are based on a first order
    division model for radial lens distortion. The radius in the undistorted
    image ru can be computed from the radius rd in the distorted image as
    follows: ru = rd / (1 + k * rd^2) where k is the distortion coefficient.
    If k < 0, barrel distortion is created. For k > 0 pincushion distortion
    occurs.
    """
    height, width, channel = image.shape
    yd, xd = np.mgrid[0:height:1, 0:width:1]
    xd = xd.astype(np.float32) - width / 2 - dx
    yd = yd.astype(np.float32) - height / 2 - dy
    theta = np.arctan2(yd, xd)
    rd = np.sqrt(xd * xd + yd * yd)
    ru = rd / (1 + k * rd * rd)
    map_x = ru * np.cos(theta) + width / 2 + dx
    map_y = ru * np.sin(theta) + height / 2 + dy
    image_distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    return image_distorted


def undistort(image, k=-400e-8, dx=0, dy=0, scaling=True):
    """Undistort a given image based on the known distortion factor and distortion center (cx, cy).
    Similar as the distort function. However it uses a negative distortion coefficient to invert the
    effect of the distortion function. The mapping function have the task to yield the coordinates
    (xd, yd) in the distorted image for a given (xu, yu) in the new undistorted image. Thus:
     -> xd = map_x(xu, yu)
     -> yd = map_y(xu, yu)
    Additionally, scaling is performed [...].
    """
    k = -k
    height, width, channel = image.shape
    yu, xu = np.mgrid[0:height:1, 0:width:1]
    xu = xu.astype(np.float32) - width / 2 - dx
    yu = yu.astype(np.float32) - height / 2 - dy
    theta = np.arctan2(yu, xu)
    ru = np.sqrt(xu * xu + yu * yu)
    rd = ru / (1 + k * ru * ru)
    if scaling:
        scaling_factor = np.max((
            ru[0, 0] / rd[0, 0],
            ru[0, -1] / rd[0, -1],
            ru[-1, 0] / rd[-1, 0],
            ru[-1, -1] / rd[-1, -1])) # find outermost corners
    else:
        scaling_factor = 1.0
    map_x = scaling_factor * rd * np.cos(theta) + width / 2 + dx
    map_y = scaling_factor * rd * np.sin(theta) + height / 2 + dy
    image_undistorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    return image_undistorted


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

    image = cv2.imread("imagenet/img.jpg")

    # crop out squre
    image = square_center_crop(image)

    # distortion center
    dx=0
    dy=0

    draw_central_rectangle(image, dw=130, dh=130, dx=dx, dy=dy, color=(0, 0, 255))
    image_distorted = distort(image, k=-500e-8, dx=dx, dy=dy)

    # crop out a region of image_distorted
    height, width, channel = image_distorted.shape
    dw=196
    dh=dw
    xmin = int(np.floor(width/2-dw+dx))
    ymin = int(np.floor(height/2-dh+dy))
    xmax = int(np.ceil(width/2+dw+dx)+1)
    ymax = int(np.ceil(height/2+dh+dy)+1)
    image_distorted_cropped = image_distorted[ymin:ymax, xmin:xmax, :]
    draw_central_rectangle(image_distorted, dw=dw, dh=dh, dx=dx, dy=dy, color=(0, 255, 0))

    # resize distorted and cropped image to original size
    image_distorted_cropped_resized = cv2.resize(image_distorted_cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)

    image_undistorted = undistort(image_distorted_cropped_resized, k=-500e-8, dx=dx, dy=dy)

    while True:
        cv2.imshow("img_original", image)
        cv2.imshow("img_distorted", image_distorted)
        cv2.imshow("image_distorted_cropped_resized", image_distorted_cropped_resized)
        cv2.imshow("image_undistorted", image_undistorted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
