import os
from os import listdir
from gte.info import IMG_DATA, IMG_FEATS
import numpy as np
from gte.images.deep_learning_models import VGG16
from keras.models import Model
from gte.utils.path import mkdir
from tqdm import tqdm
import pickle

class Image2vec(object):
    """Image2vec"""
    def __init__(self, has_model=False):
        self.image_feats = np.array([])
        self.ids = None
        if os.path.exists(IMG_FEATS):
            self._load_with_pickle()
        if not os.path.exists(IMG_FEATS) or has_model:
            vgg_full_model = VGG16(weights='imagenet')
            #We don't need the prediction layer
            self.vgg = Model(input=vgg_full_model.input, output=vgg_full_model.get_layer('block5_pool').output)

    def get_features(self, img_id):
        feats = self._lookup(img_id[:-4])
        if not feats == None:
            return feats
        return self._compute_features(img_id)

    def _compute_features(self, img_id):
        from keras.preprocessing import image
        from gte.images.deep_learning_models.imagenet_utils import preprocess_input
        img = image.load_img(IMG_DATA + "/" + img_id, target_size=(224, 224)) #[224 x 224]
        img_array = image.img_to_array(img) #[224 x 224 x channels]
        img_array = np.expand_dims(img_array, axis=0) #[1 x 224 x 224 x channels]
        #Subtract the mean RGB channels of the imagenet dataset
        #(since the model has been trained on a different dataset)
        img_array = preprocess_input(img_array) #[1 x 224 x 224 x channels]
        feats = self.vgg.predict(img_array) #[1 x 7 x 7 x 512]
        feats = np.reshape(feats, [1, 49, 512]) #[1 x 49 x 512]
        if self.image_feats.ndim == 1:
            self.image_feats = np.array(feats)
            self.ids = np.array(img_id)
        else:
            self.image_feats = np.vstack((self.image_feats, feats))
            self.ids = np.vstack((self.ids, [img_id]))
        self._save(img_id, feats[0])
        return feats[0]

    def compute_all_feats_and_store(self):
        print("Computing image features..")
        img_files = os.listdir(IMG_DATA)
        bar = tqdm(range(len(img_files)))
        for img_index in bar:
            img = img_files[img_index]
            if not (img.startswith(".") or os.path.isdir(img)) and img.endswith("jpg"):
                self.get_features(img)
                #self._compute_features(img)
        with open(IMG_FEATS + "/all.pickle", "wb") as f:
            pickle.dump(self.image_feats, f)
        with open(IMG_FEATS + "/ids.pickle", "wb") as f:
            pickle.dump(self.ids, f)
        
    def _lookup(self, img_id):
        indexes = np.where(self.ids == img_id)[0]
        return self.image_feats[indexes[0]] if indexes.size > 0 else None

    def _save(self, img, vectors):
        if not os.path.exists(IMG_FEATS):
            mkdir(IMG_FEATS)
        np.savetxt(IMG_FEATS + "/{}.txt".format(img[:-4]), vectors)

    def _load(self):
        print("Loading image features..")
        feat_files = os.listdir(IMG_FEATS)
        bar = tqdm(range(len(feat_files)))
        for feat_index in bar:
            feat_file = feat_files[feat_index]
            img_name = feat_file[:-4]
            features = np.expand_dims(np.array(np.loadtxt(IMG_FEATS + "/" + feat_file)), axis=0)
            if self.image_feats.ndim == 1:
                self.image_feats = features
                self.ids = np.array(img_name)
            else:
                self.image_feats = np.vstack((self.image_feats, features))
                self.ids = np.vstack((self.ids, [img_name]))

    def _load_with_pickle(self):
        with open(IMG_FEATS + "/all.pickle", "rb") as f:
            self.image_feats = pickle.load(f)
        with open(IMG_FEATS + "/ids.pickle", "rb") as f:
            self.ids = pickle.load(f)
