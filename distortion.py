import cv2
import numpy as np


def compute_maps(image_width, image_height, k=-0.4, dx=0, dy=0):
    """Compute distortion and undistortion maps for given distortion parameters.

    Computes a tuple of four maps which can be used to distort or undistort an
    image or sets of points with the methods below. The maps have the following
    meaning:
        maps[0] (d_map_x): Yields the undistorted x coordinate as
            a function of the distorted coordinates: xu = ud_map_x(yd, xd)
        maps[1] (d_map_y): Yields the undistorted y coordinate as
            a function of the distorted coordinates: yu = ud_map_y(yd, xd)
        maps[2] (ud_map_x): Yields the distorted x coordinate as
            a function of the undistorted coordinates: xd = d_map_x(yu, xu)
        maps[3] (ud_map_y): Yields the distorted y coordinate as
            a function of the undistorted coordinates: yd = d_map_y(yu, xu)

    Args:
        image_width (int): Image width in pixels of the image for which the
            distort map is computed.
        image_height (int): Image height in pixels.
        k (float): The distortion coefficient. Has to be smaller or equal
            to zero. Typical values lie in the range [-0.4 ... 0].
        dx (float): Offset of the distortion center in x-direction.
        dy (float): Offset of the distortion center in y-direction.

    Returns:
        maps (tuple of numpy.ndarrays): The distortion and undistortion maps
            as explained above.
    """
    assert k <= 0, "Distortion parameter k has to be zero or negative."
    w, h = image_width, image_height
    #####################################################################
    # direct radial distortion
    #####################################################################
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
        d_map_x = xd.astype(np.float32)
        d_map_y = yd.astype(np.float32)
    else:
        d_map_x = xu.astype(np.float32)
        d_map_y = yu.astype(np.float32)

    #####################################################################
    # inverse radial distortion
    #####################################################################
    yd, xd = np.mgrid[0:h, 0:w]
    if k < 0:
        # normalize coordinates to range [-0.5..0.5 x -0.5..0.5]
        xdr = xd/w - dx/w - 1/2
        ydr = yd/h - dy/h - 1/2
        # convert to polar
        rd = np.sqrt(xdr * xdr + ydr * ydr)
        theta = np.arctan2(ydr, xdr)
        # distort coordinates
        d1 = 4*k*k*rd*rd
        eps = np.finfo(float).eps  # prevents divide by zero for small d1
        d2 = 1/(d1 + eps) + 1/k
        d2[d2<0] = 0 # prevent negative value in root
        ru = -1/(2*k*rd + eps)-np.sqrt(d2)
        # convert back to cartesian
        xur = ru * np.cos(theta)
        yur = ru * np.sin(theta)
        # un-normalize coordinates to oriaginal range [0..w x 0...h]
        xu = (xur + 1/2)*w + dx
        yu = (yur + 1/2)*h + dy
        ud_map_x = xu.astype(np.float32)
        ud_map_y = yu.astype(np.float32)
    else:
        ud_map_x = xd.astype(np.float32)
        ud_map_y = yd.astype(np.float32)

    maps = (d_map_x, d_map_y, ud_map_x, ud_map_y)
    return maps


def distort_image(image, maps):
    """Distort an image based on precomputed maps.

    Args:
        image (numpy.ndarray): Input image.
        maps (tuple of numpy.ndarrays): Distortion and undistortion maps
            precomputed via compute_maps.

    Returns:
        image (numpy.ndarray): Distorted image.
    """
    image = cv2.remap(image, maps[2], maps[3], interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    return image


def undistort_image(image, maps):
    """Undistort an image based on precomputed maps.

    Args:
        image (numpy.ndarray): Input image.
        maps (tuple of numpy.ndarrays): Distortion and undistortion maps
            precomputed via compute_maps.

    Returns:
        image (numpy.ndarray): Undistorted image.
    """
    image = cv2.remap(image, maps[0], maps[1], interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    return image


def distort_points(points, maps):
    """Distort a set of points with precomputed maps.

    Args:
        points (numpy.ndarray): Points set as array of shape (N x 2) where N is
            the number of points.
        maps (tuple of numpy.ndarrays): Distortion and undistortion maps
            precomputed via compute_maps.

    Returns:
        distorted points (numpy.ndarray): Distorted point set as array with shape (N x 2).

    Note:
        This function converts the points to integers first. This reduces
        accuracy, but also leads to faster computation.
    """
    d_map_x = maps[0]
    d_map_y = maps[1]
    distorted_points = []
    for pt_x, pt_y in points:  # x, y
        pt_x_d = d_map_x[int(pt_y), int(pt_x)]
        pt_y_d = d_map_y[int(pt_y), int(pt_x)]
        distorted_points.append([pt_x_d, pt_y_d])
    return np.array(distorted_points)


def undistort_points(points, maps):
    """Undistort a set of points with precomputed maps.

    Args:
        points (numpy.ndarray): Points set as array of shape (N x 2) where N is
            the number of points.
        maps (tuple of numpy.ndarrays): Distortion and undistortion maps
            precomputed via compute_maps.

    Returns:
        undistorted points (numpy.ndarray): Undistorted point set as array with shape (N x 2).

    Note:
        This function converts the points to integers first. This reduces
        accuracy, but also leads to faster computation.
    """
    ud_map_x = maps[2]
    ud_map_y = maps[3]
    undistorted_points = []
    for pt_x, pt_y in points:  # x, y
        pt_x_d = ud_map_x[int(pt_y), int(pt_x)]
        pt_y_d = ud_map_y[int(pt_y), int(pt_x)]
        undistorted_points.append([pt_x_d, pt_y_d])
    return np.array(undistorted_points)


def crop_max(image, width, height, maps, dx, dy):
    """Crop out maximal central region from distorted image.

    Distorting an image leaves the image border unspecified (black area).
    This functions crops out the largest possible region which does not
    contain any border.

    Args:
        image (numpy.ndarray): Distorted input image of which region is to be
            cropped out.
        width (int): Width of the original undistorted image.
        height (int): Height of the original undistorted image.
        maps (tuple of numpy.ndarrays): Distortion and undistortion maps
            precomputed via compute_maps.
        dx (float): Offset of the distortion center in x-direction.
        dy (float): Offset of the distortion center in y-direction.

    Returns:
        cropped image (numpy.ndarray): Cropped version of the input image.
        coords (tuple of ints): The coordinates of the cropped
            region in the distorted input image (xd_min, yd_min, xd_max, yd_max).

    Note:
        To get the desired result, the original image with size (width x height)
        has to be distorted first before being fed into this function. The maps
        tuple used for this distortion has to be used as an argument here.
    """
    ud_map_x = maps[2]
    ud_map_y = maps[3]
    if dx >= 0 and dy >= 0:
        xd_min = ud_map_x[0, 0]  # y, x
        xd_max = ud_map_x[0, width-1]
        yd_min = ud_map_y[0, 0]
        yd_max = ud_map_y[height-1, 0]
    elif dx < 0 and dy < 0:
        xd_min = ud_map_x[height-1, 0]
        xd_max = ud_ud_map_x[height-1, width-1]
        yd_min = ud_map_y[0, width-1]
        yd_max = ud_map_y[height-1, width-1]
    elif dx >= 0 and dy < 0:
        xd_min = ud_map_x[height-1, 0]
        xd_max = ud_map_x[height-1, width-1]
        yd_min = ud_map_y[0, 0]
        yd_max = ud_map_y[height-1, 0]
    elif dx < 0 and dy >= 0:
        xd_min = ud_map_x[0, 0]
        xd_max = ud_map_x[0, width-1]
        yd_min = ud_map_y[0, width-1]
        yd_max = ud_map_y[height-1, width-1]
    xd_min = int(xd_min)
    xd_max = int(xd_max)
    yd_min = int(yd_min)
    yd_max = int(yd_max)
    coords = (xd_min, yd_min, xd_max, yd_max)
    image_cropped = image[yd_min:yd_max, xd_min:xd_max, :]
    return image_cropped, coords


def draw_central_rectangle(image, dw, dh, dx, dy, color=(255, 255, 255)):
    height, width, channel = image.shape
    xmin = int(np.floor(width/2-dw+dx))
    ymin = int(np.floor(height/2-dh+dy))
    xmax = int(np.ceil(width/2+dw+dx))
    ymax = int(np.ceil(height/2+dh+dy))
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color)
    cv2.circle(image, center=(int(width/2+dx), int(height/2+dy)), radius=10, color=color)


def draw_points(image, points, color=(255, 255, 255)):
    """Draw point(s) on the image."""
    height, width, channel = image.shape
    avg_size = (height + width) / 2
    for pt_x, pt_y in points:
        cv2.circle(image, center=(int(pt_x), int(pt_y)), radius=int(avg_size/100), thickness=-1, color=color)


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

    dx=0
    dy=0
    k=-0.4

    ##########################################################################
    # distortion & undistortion of an image
    ##########################################################################

    # prepare inpu timage
    image = cv2.imread("old/dataset_experiments/img.jpg")
    draw_central_rectangle(image, dw=130, dh=130, dx=0, dy=0, color=(0, 0, 255))

    # crop out largest possible square
    #image = square_center_crop(image)
    height, width, channel = image.shape

    # compute distortion and undistortion maps
    maps = compute_maps(width, height, k, dx, dy)

    # distort the image
    image_distorted = distort_image(image, maps)

    # undistort the distorted image
    image_undistorted = undistort_image(image_distorted, maps)

    # crop distorted image (for training)
    image_distorted_cropped, (xmin, ymin, xmax, ymax) = crop_max(image_distorted, width, height, maps, dx, dy)
    cv2.rectangle(image_distorted, (xmin, ymin), (xmax, ymax), color=(255, 255, 0))

    ##########################################################################
    # distortion & undistortion of points
    ##########################################################################

    # prepare points
    points = np.array([[100.5, 100.5],
                       [400, 100],
                       [100, 400],
                       [400, 400]])
    draw_points(image, points, color=(255, 0, 255))

    distorted_points = distort_points(points, maps)
    draw_points(image_distorted, distorted_points, color=(255, 0, 255))

    undistorted_points = undistort_points(distorted_points, maps)
    draw_points(image_undistorted, undistorted_points, color=(255, 0, 255))


    ##########################################################################
    # show images & points
    ##########################################################################

    while True:
        cv2.imshow("img_original", image)
        cv2.imshow("img_distorted", image_distorted)
        #cv2.imshow("img_distorted_cropped", image_distorted_cropped)
        cv2.imshow("image_undistorted", image_undistorted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
