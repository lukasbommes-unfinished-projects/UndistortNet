import collections
import numpy as np
import cv2
from PIL import Image
import torchvision
from torchvision import transforms

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
    average_length = np.mean(list(longest_lines.keys()))
    return longest_lines, average_length


if __name__ == "__main__":
    image_name = "imagenet/img.jpg"
    # load as PIL image
    with open(image_name, "rb") as img_file:
        im = Image.open(img_file)
        im = im.convert('RGB')
    # resize to 256 px (smaller edge) and crop out central 256 x 256 region
    transform = transforms.Compose([
       transforms.Resize(512),
       transforms.CenterCrop(512)
    ])
    im = transform(im)
    # convert to openCV image
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2GRAY)
    #cv2.imshow("source", im)

    #im = cv2.GaussianBlur(im, (5, 5), 0, 0)
    im = cv2.blur(im, (2, 2))
    #cv2.imshow("im_blur", im)
    #cv2.waitKey(500)

    # ret, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imshow("im_threshold", im)
    # cv2.waitKey(500)

    # kernel=(5, 3)
    # for _ in range(5):
    #     im = cv2.dilate(im, kernel=kernel)
    #     im = cv2.erode(im, kernel=kernel)
    # for _ in range(5):
    #     im = cv2.erode(im, kernel=kernel)
    #     im = cv2.dilate(im, kernel=kernel)
    # cv2.imshow("im_morphological", im)
    # cv2.waitKey(500)

    im = cv2.Canny(im, 50, 200, None, 3)
    #cv2.imshow("im_canny", im)
    #cv2.waitKey(500)

    # contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("im_contour", im)
    # cv2.waitKey(500)

    # im = cv2.Canny(im, 50, 200, None, 3)
    # cv2.imshow("im_canny_2", im)
    # cv2.waitKey(500)

    c_im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    linesP = cv2.HoughLinesP(im, 1, np.pi / 180, 75, None, 15, 5)

    longest_lines, average_length = get_k_longest_lines(linesP, k=20)
    print(average_length)
    for l in longest_lines.values():
        cv2.line(c_im, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 1, cv2.LINE_AA)

    alpha = 0.2
    if average_length >= alpha:
        print("image selected")

    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(c_im, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)

    while True:

        #cv2.imshow("detected lines", c_im)
        cv2.imshow("detected lines (prob)", c_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
