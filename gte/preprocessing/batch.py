import numpy as np
import math
import csv

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from gte.info.info import MAX_LEN_P, MAX_LEN_H, UNK, PAD
from gte.utils.dic import dic_lookup_case_sensitive

class Batch(object):
    """Batch"""
    def __init__(self, batch_size, P, H, I, IDs, labels, word2id, label2id, max_len_p=MAX_LEN_P, max_len_h=MAX_LEN_H):
        self.size = batch_size
        lookup = lambda x: dic_lookup_case_sensitive(word2id, x, UNK)
        self.P = self._map_sequences_id(P, lookup, max_len_p)
        self.H = self._map_sequences_id(H, lookup, max_len_h)
        self.labels = np.array([label2id[label] for label in labels])
        self.lengths_P = np.array([len(p) for p in P])
        self.lengths_H = np.array([len(h) for h in H])
        self.IDs = np.array(IDs)

        assert len(self.labels) == len(self.P)
        assert len(self.labels) == len(self.H)
        assert len(self.labels) == self.size

    def _map_sequences_id(self, sequences, lookup, maxlen):
        # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
        sequences = [list(map(lookup, sequence)) for sequence in sequences]
        return pad_sequences(sequences, maxlen=maxlen, dtype='int32', padding='post', truncating='post', value=PAD)


# use a generator
def generate_batch(dataset_file, batch_size, word2id, label2id, max_len_p=MAX_LEN_P, max_len_h=MAX_LEN_H):
    # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
    with open(dataset_file) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None) #skip header
        P, H, labels, I, IDs = [], [], [], [], []
        last_batch = False
        while not last_batch:
            while len(labels) < batch_size:
                row = next(reader, None)
                if row == None:
                    #last batch is not complete
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
                    # non token not used
                    # premise = row[4].strip()
                    # hypothesis = row[5].strip()
            #complete batch
            batch = Batch(batch_size, P, H, I, IDs, labels, word2id, label2id, max_len_p=MAX_LEN_P, max_len_h=MAX_LEN_H)
            yield batch

def iteration_per_epoch(dataset_file, batch_size):
    with open(dataset_file) as f:
           return math.ceil(len(list(f)) / batch_size)
