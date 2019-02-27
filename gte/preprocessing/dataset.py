import os
import random
import json
import csv
import string
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


def datasets_to_word_set_VE():
    datasets = [TRAIN_DATA, DEV_DATA, TEST_DATA]
    translation = {ord(char): " {}".format(char) for char in string.punctuation}

    words = []
    labels = set()
    lens_p = []
    lens_h = []
    for filename in datasets:
        with open(filename) as in_file:
            for line in tqdm(in_file.readlines()):
                row = json.loads(line)
                label = str(row['gold_label'])
                labels.add(label)

                premise_tokens = str(row['sentence1']).strip().translate(translation).split()
                lens_p += [len(premise_tokens)]
                words.extend(sentence_to_words(premise_tokens))

                hypothesis_tokens = str(row['sentence2']).strip().translate(translation).split()
                lens_h += [len(hypothesis_tokens)]
                words.extend(sentence_to_words(hypothesis_tokens))

    return set(words), labels, lens_p, lens_h


def sentence_to_words(sentence):
    return { w for w in sentence }.union({ w.lower() for w in sentence  })

def words_to_dictionary(words, embedding_name, embedding_size):
    # dic, inv_dic = index_map(list(words), unk=UNK)
    embeddings, word_to_index, index_to_word = retrieve_embeddings(embedding_name, embedding_size, words)
    return embeddings, word_to_index, index_to_word

def generate_non_token_datasets():
    datasets = [TRAIN_DATA, DEV_DATA, TEST_DATA, TEST_DATA_HARD]
    TO_BE_TAGGED_DIR = './DATA/vsnli/TO_BE_TAGGED'
    mkdir(TO_BE_TAGGED_DIR)
    for filename in datasets:
        P = ""
        H = ""
        with open(filename) as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None) #skip header
            for row in reader:
                P += (row[4].strip() + "\n")
                H += (row[5].strip() + "\n")
        name = filename.split("/")[-1][:-4]
        with open(TO_BE_TAGGED_DIR + "/premises_{}.txt".format(name), "w+") as f:
            f.write(P)
        with open(TO_BE_TAGGED_DIR + "/hypothesis_{}.txt".format(name), "w+") as f:
            f.write(H)

#Format: LP1_LP2_LP3#LH1_LH2_LH3\tDP1_DP2_DP3#DH1_DH2_DH3
def generate_datasets_with_dependency():
    datasets = [TRAIN_DATA, DEV_DATA, TEST_DATA, TEST_DATA_HARD]
    DEP_DIR = './DATA/vsnli/DEP'
    TAGGED_DIR = './DATA/vsnli/TAGGED'
    mkdir(DEP_DIR)
    for filename in datasets:
        new_dataset = ""
        name = filename.split("/")[-1][:-4]
        P_file = TAGGED_DIR + "/premises_{}.txt".format(name)
        H_file = TAGGED_DIR + "/hypothesis_{}.txt".format(name)
        with open(filename) as f, open(P_file) as P_f, open(H_file) as H_f:
            lines = f.readlines()[1:]#skip header
            P_reader = csv.reader(P_f, delimiter="\t")
            H_reader = csv.reader(H_f, delimiter="\t")
            for i,row in enumerate(lines): #for each sentence
                levels = ""
                relations = ""
                for P_word in P_reader: #for each word in P
                    if not P_word:
                        break
                    levels += P_word[6] + "_"
                    relations += P_word[7] + "_"
                levels = levels[:-1] + "#"
                relations = relations[:-1] + "#"
                for H_word in H_reader: #for each word in P
                    if not H_word:
                        break
                    levels += H_word[6] + "_"
                    relations += H_word[7] + "_"
                levels = levels[:-1]
                relations = relations[:-1]
                new_dataset += row.strip("\n") + "\t" + levels + "\t" + relations + "\n"
        with open(DEP_DIR + "/{}.tsv".format(name), "w+") as f:
            f.write(new_dataset)


# def datasets_to_index(datasets, word_to_index, use_only_token=True):

def datasets_to_dep_set():
    datasets = [TRAIN_DATA, DEV_DATA, TEST_DATA, TEST_DATA_HARD]

    dep = set()
    for filename in datasets:
        with open(filename) as in_file:
            reader = csv.reader(in_file, delimiter="\t")
            next(reader, None) #skip header
            for row in tqdm(reader):
                # l = row[LABEL].strip()
                relations_P = set(row[8].strip().split("#")[0].split('_'))
                relations_H = set(row[8].strip().split("#")[1].split('_'))
                dep = dep | relations_P | relations_H
    return dep

def _load_stopwords():
    stopwords = set()
    filename = "/home/agostina/master/NLP/hw/NLP/hw1/stopwords_en.txt"
    with open(filename) as f:
        for line in f.readlines():
            stopwords.add(line.strip())
    return stopwords

def dataset_without_stopwords():
    datasets = [TRAIN_DATA, DEV_DATA, TEST_DATA, TEST_DATA_HARD]
    stopwords = _load_stopwords()
    DEP_DIR = './DATA/vsnli/DEP'
    NO_STOPWORDS_DIR = DEP_DIR + "/NO_STOPWORDS"
    mkdir(NO_STOPWORDS_DIR)
    for filename in datasets:
        name = filename.split("/")[-1][:-4]
        with open(filename) as in_file, open(NO_STOPWORDS_DIR + "/" + name + ".tsv", "w+") as out_file:
            reader = csv.reader(in_file, delimiter="\t")
            writer = csv.writer(out_file, delimiter="\t")
            header = next(reader, None)
            writer.writerow(header)
            for row in reader:
                P = row[1].strip().split()
                H = row[2].strip().split()             
                levels = row[7].strip().split("#")
                P_level = levels[0].split("_")
                H_level = levels[1].split("_")
                relations = row[8].strip().split("#")
                P_rel = relations[0].split("_")
                H_rel = relations[1].split("_")
                to_remove = []
                for index,word in enumerate(P):
                    if word.lower() in stopwords:
                        to_remove += [index]
                P = ' '.join([i for j, i in enumerate(P) if j not in to_remove])
                P_level = '_'.join([i for j, i in enumerate(P_level) if j not in to_remove])
                P_rel = '_'.join([i for j, i in enumerate(P_rel) if j not in to_remove])
                to_remove = []
                for index,word in enumerate(H):
                    if word.lower() in stopwords:
                        to_remove += [index]
                H = ' '.join([i for j, i in enumerate(H) if j not in to_remove])
                H_level = '_'.join([i for j, i in enumerate(H_level) if j not in to_remove])
                H_rel = '_'.join([i for j, i in enumerate(H_rel) if j not in to_remove])
                row[1] = P
                row[2] = H
                row[7] = P_level + "#" + H_level
                row[8] = P_rel + "#" + H_rel
                writer.writerow(row)