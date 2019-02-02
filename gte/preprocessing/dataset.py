import os
import random
import csv
from tqdm import tqdm
from gte.info import TRAIN_DATA, DEV_DATA, TEST_DATA, TEST_DATA_HARD, X_TRAIN_DATA, X_DEV_DATA, X_TEST_DATA, X_TEST_DATA_HARD, UNK, SHUFFLED_DIR
from gte.utils.dic import index_map
from gte.emb.emb import retrieve_embeddings
from gte.utils.path import mkdir

LABEL = 0
PREMISE_TOKEN = 1
HYPOTHESIS_TOKENS = 2
PREMISE = 4
HYPOTHESIS = 5

def generate_shuffled_datasets():
    datasets = [X_TRAIN_DATA, X_DEV_DATA, X_TEST_DATA, X_TEST_DATA_HARD]
    SHUFFLED_DIR = './DATA/vsnli/SHUFFLED'
    mkdir('./DATA/vsnli/SHUFFLED')
    for filename in datasets:
        # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
        lines = open(filename).readlines()
        header = lines[0]
        lines = lines[1:]
        random.shuffle(lines)
        name = os.path.basename(filename)
        shuffled_file = os.path.join(SHUFFLED_DIR, name)
        open(shuffled_file, 'w').writelines([header] + lines)

def datasets_to_word_set(use_only_token=True):
    datasets = [TRAIN_DATA, DEV_DATA, TEST_DATA, TEST_DATA_HARD]

    words = []
    labels = set()
    lens_p = []
    lens_h = []
    for filename in datasets:
        with open(filename) as in_file:
            reader = csv.reader(in_file, delimiter="\t")
            next(reader, None) #skip header
            for row in tqdm(reader):
                label = row[LABEL].strip()
                labels.add(label)

                premise_tokens = row[PREMISE_TOKEN].strip().split()
                lens_p += [len(premise_tokens)]
                words.extend(sentence_to_words(premise_tokens))

                hypothesis_tokens = row[HYPOTHESIS_TOKENS].strip().split()
                lens_h += [len(hypothesis_tokens)]
                words.extend(sentence_to_words(hypothesis_tokens))

                if not use_only_token:
                    premise = row[PREMISE].strip()
                    words.extend(sentence_to_words(premise))

                    hypothesis = row[HYPOTHESIS].strip()
                    words.extend(sentence_to_words(hypothesis))
    return set(words), labels, lens_p, lens_h

def sentence_to_words(sentence):
    return { w for w in sentence }.union({ w.lower() for w in sentence  })

def words_to_dictionary(words, embedding_name, embedding_size):
    # dic, inv_dic = index_map(list(words), unk=UNK)
    embeddings, word_to_index, index_to_word = retrieve_embeddings(embedding_name, embedding_size, words)
    return embeddings, word_to_index, index_to_word

# def datasets_to_index(datasets, word_to_index, use_only_token=True):

