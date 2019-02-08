import os
import requests
import urllib

from tqdm import *
from bs4 import BeautifulSoup
import numpy as np
import PIL.Image
import cv2


train_dir = "train"
val_dir = "val"

my_synsets = {"city, metropolis": "n08226335",
            "town, townspeople, townsfolk": "n08226514"}
            #"village, small town, settlement": "n08226699",
            #"street": "n08225426",
            #"automobile factory, auto factory, car factory": "n02761696",
            #"factory, mill, manufacturing plant, manufactory": "n03316406",
            #"steel mill, steelworks, steel plant, steel factory": "n04311595",
            #"fabrication, manufacture, manufacturing": "n00924825",
            #"machine tool": "n03702248",
            #"traffic lane": "n04466386",
            #"architecture": "n06123363"}

def get_urls_for_synsets(synsets):
    image_urls = {}
    for words, wnid in synsets.items():
        # get url of all images in the synset with wnid
        page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}".format(wnid))
        soup = BeautifulSoup(page.content, "html.parser")
        str_soup = str(soup)
        url_list = str_soup.split('\r\n')
        print("Number of images in synset \"{}\" (wnid: {}): {}".format(words, wnid, len(url_list)))
        print(url_list)
        image_urls[wnid] = url_list
    return image_urls


def get_image_from_url(url):
    """Download an image from url and return it as numpy array."""
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


# download images

image_urls = get_urls_for_synsets(my_synsets)

print("Started downloading...")
for i, (wnid, url_list) in enumerate(image_urls.items()):
    print("Downloading {} images of synset {}...")
    for url in url_list:
        image = get_image_from_url(url)
        try:
            if len(image.shape) == 3:
                save_path = os.path.join(train_dir, wnid, "{}.jpg".format(i))
                cv2.imwrite(save_path, image)
        except (urllib.error.URLError, urllib.error.HTTPError):
            raise #continue



# create the test set
