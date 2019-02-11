import os
import requests
import urllib

from tqdm import *
from bs4 import BeautifulSoup
import numpy as np
import PIL.Image
import cv2


output_dir = "images"

my_synsets = {"house": "n03544360",
              "house_2": "n03545150"}

            # "city, metropolis": "n08226335",
            # "town, townspeople, townsfolk": "n08226514"}
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
        print("{}: \"{}\" ({} images)".format(wnid, words, len(url_list)))
        image_urls[wnid] = url_list
    return image_urls


def get_image_from_url(url):
    """Download an image from url and return it as numpy array."""
    resp = urllib.request.urlopen(url, timeout=2)  # timeout 5 seconds
    # check if there was a redirect to a new url (e.g. because image is not available)
    new_url = resp.geturl()
    if new_url == url:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        image = None
    return image


def filter_out_existing_synsets(dir, synsets):
    if os.path.exists(dir):
        existing_synsets = os.listdir(dir)
        output_synsets = {}
        for words, wnid in synsets.items():
            if wnid not in existing_synsets:
                output_synsets[words] = wnid
        return output_synsets
    else:
        return synsets


if __name__ == "__main__":

    my_synsets = filter_out_existing_synsets(output_dir, my_synsets)
    if len(my_synsets) > 0:
        print("The following synsets will be downloaded")
    else:
        print("No synsets to download. Exiting...")

    image_urls = get_urls_for_synsets(my_synsets)
    for wnid, url_list in image_urls.items():
        print("Downloading images of synset {}".format(wnid))
        try:
            with tqdm(total=len(url_list)) as pbar:
                for i, url in enumerate(url_list):
                    try:
                        image = get_image_from_url(url)
                        if image is not None:
                            if len(image.shape) == 3:
                                save_path = os.path.join(output_dir, wnid)
                                if not os.path.exists(save_path):
                                    os.makedirs(save_path)
                                file_name = os.path.join(save_path, "{:08d}.jpg".format(i))
                                cv2.imwrite(file_name, image)
                        pbar.update(1)
                    except KeyboardInterrupt:
                        raise
                    except: # (urllib.error.URLError, urllib.error.HTTPError):
                        continue

        except KeyboardInterrupt:
            break
