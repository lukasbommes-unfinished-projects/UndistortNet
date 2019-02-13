import os
import glob
import collections
import shutil
import tarfile
import cv2
import numpy as np
from PIL import Image
from tqdm import *
import torchvision
from torchvision import transforms


def detect_edges(image):
    image = cv2.blur(image, (2, 2))
    image = cv2.Canny(image, 50, 200, None, 3)
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 75, None, 15, 5)
    return image, lines


def draw_edges(image, lines):
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(color_image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)
    return color_image


def get_k_longest_lines(lines, k=10):
    line_lenghts = {}
    if lines is not None:
        for line in lines:
            l = line[0]
            line_lenght = np.sqrt(np.square(l[2]-l[0]) + np.square(l[3]-l[1]))
            line_lenghts[line_lenght] = l
    sorted_line_lenghts = collections.OrderedDict(sorted(line_lenghts.items(), reverse=True))
    i = 0
    longest_lines = {}
    for length, line_coordinates in sorted_line_lenghts.items():
        longest_lines[length] = line_coordinates
        i += 1
        if i == k:
            break
    if len(longest_lines) > 0:
        average_length = np.mean(list(longest_lines.keys()))
    else:
        average_length = 0
    return longest_lines, average_length


if __name__ == "__main__":
    """This script filters copies images from the input_dir into the output_dir
    if the following criterium is met:
    - The average lenghts of the top_k longest lines has to be
      greater or equal to alpha*im_size.
    Lines are found via canny edge detection followed by probabilistic Hough
    transform. The line lenghts are computed, sorted and the top_k longest lines
    are retrieved and their average length is computed. A copy-command is issued
    when the criterium is met.
    """

    im_size = 256#512
    alpha = 0.7
    top_k = 3

    file_groups = ["val", "test", "train"]
    for file_group in file_groups:

        if file_group == "test":
            input_dir = os.path.join("ILSVRC2012_img_test", "*.JPEG")
            output_dir = "distortion_dataset/test"
        elif file_group == "val":
            input_dir = os.path.join("ILSVRC2012_img_val", "*.JPEG")
            output_dir = "distortion_dataset/val"
        elif file_group == "train":
            input_dir = os.path.join("ILSVRC2012_img_train/**/", "*.JPEG")
            output_dir = "distortion_dataset/train"

        # iterate over all images and search for those meeting criteria
        image_names = glob.glob(input_dir)
        for image_name in tqdm(image_names, desc=file_group):
            # load as PIL image
            with open(image_name, "rb") as img_file:
                try:
                    im = Image.open(img_file)
                except IOError:
                    break
                im = im.convert('RGB')
            # resize to 256 px (smaller edge) and crop out central 256 x 256 region
            transform = transforms.Compose([
                transforms.Resize(im_size),
                transforms.CenterCrop(224)
            ])
            im = transform(im)
            # convert to openCV image
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
            im_original = im.copy()

            im, lines = detect_edges(im)
            im = draw_edges(im, lines)

            longest_lines, average_length = get_k_longest_lines(lines, k=top_k)
            #print(average_length)
            if len(longest_lines) > 0:
                for l in longest_lines.values():
                    cv2.line(im, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1, cv2.LINE_AA)

                # filter out images with too small average line length
                if average_length >= alpha*im_size:
                    if file_group == "train":
                        folder_name = image_name.split(os.sep)[-2]
                        dst_path = os.path.join(output_dir, folder_name)
                        dst = os.path.join(dst_path, os.path.basename(image_name))
                        if not os.path.exists(dst_path):
                            os.makedirs(dst_path)
                    else:
                        dst = os.path.join(output_dir, os.path.basename(image_name))
                    shutil.copyfile(src=image_name, dst=dst)

                #cv2.imshow("im_original", im_original)
                #cv2.imshow("im_edges", im)

                #while True:
                #    if cv2.waitKey(10) & 0xFF == ord('q'):
                #        break

#cv2.destroyAllWindows()
