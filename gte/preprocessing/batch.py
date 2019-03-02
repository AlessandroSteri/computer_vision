import numpy as np
import math
import csv
import os
import string
import json

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image

from gte.info.info import MAX_LEN_P, MAX_LEN_H, UNK, PAD, IMG_DATA, HEIGHT, WIDTH, CHANNELS
from gte.utils.dic import dic_lookup_case_sensitive

class Batch(object):
    """Batch"""
    def __init__(self, batch_size, P, H, I, IDs, labels, word2id, label2id, max_len_p, max_len_h, rel2id, labelid2id=None, P_lv=None, H_lv=None, P_rel=None, H_rel=None, full_img=False):
        self.size = batch_size
        lookup = lambda x: dic_lookup_case_sensitive(word2id, x, UNK)
        self.P = self._map_sequences_id(P, lookup, max_len_p)
        self.H = self._map_sequences_id(H, lookup, max_len_h)
        if not labelid2id:
            self.labels = np.array([label2id[label] for label in labels])
        else:
            self.labels = np.array([labelid2id[label2id[label]] for label in labels])
        self.lengths_P = np.array([min(len(p), max_len_p) for p in P])
        self.lengths_H = np.array([min(len(h), max_len_h) for h in H])
        self.IDs = np.array(IDs)
        self.I = I

        if full_img:
            import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
            def id_to_img(img_id):
                img = image.load_img(IMG_DATA + "/" + img_id, target_size=(256, 256)) #[256 x 256]
                img_array = image.img_to_array(img) #[256 x 256 x channels]
                return np.expand_dims(img_array, axis=0) #[1 x 256 x 256 x channels]
            self.IMG = [id_to_img(iid) for iid in I]
        if P_lv is not None:
            self.P_lv = pad_sequences(P_lv, maxlen=max_len_p, dtype='int32', padding='post', truncating='post', value=PAD)
            self.H_lv = pad_sequences(H_lv, maxlen=max_len_h, dtype='int32', padding='post', truncating='post', value=PAD)
            rel_lookup = lambda x: rel2id[x.lower()] if x else rel2id['empty']
            self.P_rel = self._map_sequences_id(P_rel, rel_lookup, max_len_p)
            self.H_rel = self._map_sequences_id(H_rel, rel_lookup, max_len_h)

            assert len(self.labels) == len(self.P_lv)
            assert len(self.labels) == len(self.H_lv)
            assert len(self.labels) == len(self.P_rel)
            assert len(self.labels) == len(self.H_rel)

        assert len(self.labels) == len(self.P)
        assert len(self.labels) == len(self.H)
        assert len(self.labels) == self.size

    def _map_sequences_id(self, sequences, lookup, maxlen):
        # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
        sequences = [list(map(lookup, sequence)) for sequence in sequences]
        return pad_sequences(sequences, maxlen=maxlen, dtype='int32', padding='post', truncating='post', value=PAD)


def generate_batch(dataset_file, batch_size, word2id, label2id, rel2id, labelid2id=None, img2vec=None, max_len_p=MAX_LEN_P, max_len_h=MAX_LEN_H, with_DEP=False, dataset="VESNLI"):
    if dataset == "VESNLI":
        return generate_batch_vesnli(dataset_file, batch_size, word2id, label2id, rel2id, labelid2id, img2vec, max_len_p, max_len_h, with_DEP)
    else:
        return generate_batch_snli(dataset_file, batch_size, word2id, label2id, rel2id, img2vec, max_len_p, max_len_h, with_DEP)

# use a generator
def generate_batch_snli(dataset_file, batch_size, word2id, label2id, rel2id, img2vec=None, max_len_p=MAX_LEN_P, max_len_h=MAX_LEN_H, with_DEP=False):
    # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
    print("Old dataset")
    with open(dataset_file) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None) #skip header

        last_batch = False
        end_epoch = False
        while not last_batch:
            if end_epoch:
                batch = None
            else:
                P, H, labels, I, IDs, P_lv, H_lv, P_rel, H_rel = [], [], [], [], [], [], [], [], []
                while len(labels) < batch_size:
                    row = next(reader, None)
                    if row == None:
                        #last batch is not complete
                        f.seek(0)
                        reader = csv.reader(f, delimiter="\t")
                        next(reader, None) #skip header
                        last_batch = True
                    else:
                        # batch_txt += [row]
                        labels += [row[0].strip()]
                        P   += [row[1].strip().split()]
                        H   += [row[2].strip().split()]
                        img  = row[3].strip().split("#")
                        ID   = row[6].strip().split("#")[1]
                        I   += [img]
                        IDs += ID
                        if with_DEP:
                            levels = row[7].strip().split("#")
                            P_level = levels[0].split("_")
                            if not P_level[0]: #If parsing failed on P
                                P_level = [max_len_p-1] * max_len_p #Set all words to the lowest level
                            P_lv += [P_level]
                            H_level = levels[1].split("_")
                            if not H_level[0]: #If parsing failed on H
                                H_level = [max_len_h-1] * max_len_h #Set all words to the lowest level
                            H_lv += [H_level]
                            relations = row[8].strip().split("#")
                            P_rel += [relations[0].split("_")]
                            H_rel += [relations[1].split("_")]

                        # non token not used
                        # premise = row[4].strip()
                        # hypothesis = row[5].strip()
                #complete batch
                if img2vec == None:
                # if full_img:
                    # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
                    def id_to_img(img_id):
                        img = image.load_img(IMG_DATA + "/" + img_id, target_size=(256, 256)) #[256 x 256]
                        img_array = image.img_to_array(img) #[256 x 256 x channels]
                        return np.expand_dims(img_array, axis=0) #[1 x 256 x 256 x channels]
                    I = [id_to_img(iid[0]) for iid in I]
                    I = np.reshape(I, (batch_size, WIDTH, HEIGHT, CHANNELS))
                    # I = np.ones([batch_size, 49, 512], dtype=np.float32)
                    # print("[WARNING] NO VGG FEATURE FOR IMAGES.")
                else:
                    I = np.array([img2vec.get_features(i[0]) for i in I])
                if not with_DEP:
                    P_lv = None
                batch = Batch(batch_size, P, H, I, IDs, labels, word2id, label2id, max_len_p, max_len_h, rel2id, P_lv, H_lv, P_rel, H_rel)
            yield batch
            end_epoch = last_batch
            last_batch = False


def generate_batch_vesnli(dataset_file, batch_size, word2id, label2id, rel2id, labelid2id=None, img2vec=None, max_len_p=MAX_LEN_P, max_len_h=MAX_LEN_H, with_DEP=False):
    translation = {ord(char): " {}".format(char) for char in string.punctuation}
    with open(dataset_file) as f:
        last_batch = False
        end_epoch = False
        while not last_batch:
            if end_epoch:
                batch = None
            else:
                P, H, labels, I, IDs, P_lv, H_lv, P_rel, H_rel = [], [], [], [], [], [], [], [], []
                while len(labels) < batch_size:
                    row = f.readline()
                    if not row:
                        #last batch is not complete
                        f.seek(0)
                        row = f.readline()
                        line = json.loads(row)
                        last_batch = True
                    else:
                        line = json.loads(row)
                        labels += [str(line['gold_label'])]
                        H += [str(line['sentence2']).strip().translate(translation).split()]
                        img = str(line['Flikr30kID']).strip().split("#")
                        I += [img]
                        P += [str(line['sentence1']).strip().translate(translation).split()]
                        ID = str(line['pairID']).strip().split("#")[1]
                        IDs += ID
    
                #complete batch
                if img2vec == None:
                # if full_img:
                    # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
                    def id_to_img(img_id):
                        img = image.load_img(IMG_DATA + "/" + img_id, target_size=(256, 256)) #[256 x 256]
                        img_array = image.img_to_array(img) #[256 x 256 x channels]
                        return np.expand_dims(img_array, axis=0) #[1 x 256 x 256 x channels]
                    I = [id_to_img(iid[0]) for iid in I]
                    I = np.reshape(I, (batch_size, WIDTH, HEIGHT, CHANNELS))
                    # I = np.ones([batch_size, 49, 512], dtype=np.float32)
                    # print("[WARNING] NO VGG FEATURE FOR IMAGES.")
                else:
                    I = np.array([img2vec.get_features(i[0]) for i in I])
                if not with_DEP:
                    P_lv = None
                batch = Batch(batch_size, P, H, I, IDs, labels, word2id, label2id, max_len_p, max_len_h, rel2id, labelid2id, P_lv, H_lv, P_rel, H_rel)
            yield batch
            end_epoch = last_batch
            last_batch = False


def iteration_per_epoch(dataset_file, batch_size):
    with open(dataset_file) as f:
           return math.ceil(len(list(f)) / batch_size)
