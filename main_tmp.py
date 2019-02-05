import pickle
import os
import numpy as np
from gte.info.info import IMG_DATA, IMG_FEATS
from tqdm import tqdm

def main():
    num_img = 0
    ids = []
    # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
    print("Computing image features..")
    img_files = os.listdir(IMG_DATA)
    bar = tqdm(range(len(img_files)))
    for img_index in bar:
        img = img_files[img_index]
        if not (img.startswith(".") or os.path.isdir(img)) and img.endswith("jpg"):
            num_img += 1
            ids += [img]
            pass
    ids = np.array(ids)
    assert len(ids) == num_img
    image_feats = np.ones([len(ids), 49, 512])
    with open(IMG_FEATS + "/all.pickle", "wb") as f:
        pickle.dump(image_feats, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(IMG_FEATS + "/ids.pickle", "wb") as f:
        pickle.dump(ids, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
