import os
import pickle
from tqdm import *

from distortion import compute_maps


def compute_maps_for_param_ranges(dir, image_width, image_height, ks, dx, dy):
    """Precompute distortion and undistortion maps for different distortion parameters."""
    if not os.path.isdir(dir):
        os.mkdir(dir)
    print("Computing sets of maps for {} parameter combinations.".format(len(ks)*len(dxs)*len(dys)))
    for k in tqdm(ks):
        for dx in dxs:
            for dy in dys:
                maps = compute_maps(image_width, image_height, k, dx, dy)
                fname = os.path.join(dir, "maps_{}_{}_{}.pkl".format(str(k), str(dx), str(dy)))
                pickle.dump(maps, open(fname, "wb"))


if __name__ == "__main__":

    # specifiy parameter ranges
    ks = [-4*x*1e-3 for x in range(0, 101, 5)]
    dxs = list(range(-50, 51, 5))
    dys = list(range(-50, 51, 5))

    compute_maps_for_param_ranges("distortion_maps", 512, 512, ks, dxs, dys)
