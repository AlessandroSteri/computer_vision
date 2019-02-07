import numpy as np
import math
import csv

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from gte.info.info import MAX_LEN_P, MAX_LEN_H, UNK, PAD
from gte.utils.dic import dic_lookup_case_sensitive

class Batch(object):
    """Batch"""
    def __init__(self, batch_size, P, H, I, IDs, labels, word2id, label2id, max_len_p, max_len_h, rel2id, P_lv=None, H_lv=None, P_rel=None, H_rel=None):
        self.size = batch_size
        lookup = lambda x: dic_lookup_case_sensitive(word2id, x, UNK)
        self.P = self._map_sequences_id(P, lookup, max_len_p)
        self.H = self._map_sequences_id(H, lookup, max_len_h)
        self.labels = np.array([label2id[label] for label in labels])
        self.lengths_P = np.array([min(len(p), max_len_p) for p in P])
        self.lengths_H = np.array([min(len(h), max_len_h) for h in H])
        self.IDs = np.array(IDs)
        self.I = I
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


# use a generator
def generate_batch(dataset_file, batch_size, word2id, label2id, rel2id, img2vec=None, max_len_p=MAX_LEN_P, max_len_h=MAX_LEN_H, with_DEP=False):
    # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
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
                            P_lv += [levels[0].split("_")]
                            H_lv += [levels[1].split("_")]
                            relations = row[8].strip().split("#")
                            P_rel += [relations[0].split("_")]
                            H_rel += [relations[1].split("_")]

                        # non token not used
                        # premise = row[4].strip()
                        # hypothesis = row[5].strip()
                #complete batch
                if img2vec == None:
                    I = np.ones([batch_size, 49, 512], dtype=np.float32)
                else:
                    I = np.array([img2vec.get_features(i[0]) for i in I])
                batch = Batch(batch_size, P, H, I, IDs, labels, word2id, label2id, max_len_p, max_len_h, rel2id, P_lv, H_lv, P_rel, H_rel)
            yield batch
            end_epoch = last_batch
            last_batch = False

def iteration_per_epoch(dataset_file, batch_size):
    with open(dataset_file) as f:
           return math.ceil(len(list(f)) / batch_size)
