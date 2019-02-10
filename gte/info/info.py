TB_DIR = './log/tensorboard'
HP_LOG = './log/log.txt'
UNK = "<UNK>"

# NON SHUFFLED, FOLLOWS PATTERN
X_TRAIN_DATA = './DATA/vsnli/VSNLI_1.0_train.tsv'
X_DEV_DATA = './DATA/vsnli/VSNLI_1.0_dev.tsv'
X_TEST_DATA = './DATA/vsnli/VSNLI_1.0_test.tsv'
X_TEST_DATA_HARD = './DATA/vsnli/VSNLI_1.0_test_hard.tsv'

SHUFFLED_DIR = './DATA/vsnli/SHUFFLED'

TRAIN_DATA = './DATA/vsnli/DEP/VSNLI_1.0_train.tsv'
DEV_DATA = './DATA/vsnli/DEP/VSNLI_1.0_dev.tsv'
TEST_DATA = './DATA/vsnli/DEP/VSNLI_1.0_test.tsv'
TEST_DATA_HARD = './DATA/vsnli/DEP/VSNLI_1.0_test_hard.tsv'
TEST_DATA_DEMO = './DATA/vsnli/DEP/VSNLI_1.0_demo.tsv'

#TRAIN_DATA = './DATA/vsnli/DEP/NO_STOPWORDS/VSNLI_1.0_train.tsv'
#DEV_DATA = './DATA/vsnli/DEP/NO_STOPWORDS/VSNLI_1.0_dev.tsv'
#TEST_DATA = './DATA/vsnli/DEP/NO_STOPWORDS/VSNLI_1.0_test.tsv'
#TEST_DATA_HARD = './DATA/vsnli/DEP/NO_STOPWORDS/VSNLI_1.0_test_hard.tsv'

IMG_DATA = './DATA/flickr30k_images'
IMG_FEATS = './DATA/flickr30k_images/features'

BEST_F1 = './log/f1.txt'
BEST_MODEL = './log/BEST_MODEL'

MAX_LEN_P = 82
MAX_LEN_H = 62

PAD = 0

NUM_CLASSES = 3

EPS = 1e-6

NUM_FEATS = 49
FEAT_SIZE = 512

LEN_TRAIN = 545621 - 1
LEN_DEV = 9843 - 1
LEN_TEST = 9825 - 1
LEN_TEST_HARD = 545621 - 1
DEP_REL = {'empty',
           'cop',
           'ccomp',
           'advcl',
           'predet',
           'rcmod',
           'number',
           'tmod',
           'nsubjpass',
           'auxpass',
           'punct',
           'dobj',
           'iobj',
           'aux',
           'goeswith',
           'nsubj',
           'infmod',
           'expl',
           'poss',
           'neg',
           'partmod',
           'mark',
           'quantmod',
           'csubj',
           'det',
           'mwe',
           'discourse',
           'acomp',
           'cc',
           'possessive',
           'pcomp',
           'pobj',
           'ROOT',
           'prep',
           'preconj',
           'prt',
           'xcomp',
           'dep',
           'nn',
           'amod',
           'appos',
           'parataxis',
           'conj',
           'npadvmod',
           'num',
           'advmod'}

DEP_REL = {d.lower() for d in DEP_REL}
DEP_REL_SIZE = len(DEP_REL)

# DEP_REL = [
#     "acl",
#     "acomp",
#     "advcl",
#     "advmod",
#     "amod",
#     "appos",
#     "aux",
#     "case",
#     "cc",
#     "ccomp",
#     "clf",
#     "compound",
#     "conj",
#     "cop",
#     "csubj",
#     "dep",
#     "det",
#     "discourse",
#     "dislocated",
#     "dobj",
#     "expl",
#     "fixed",
#     "flat",
#     "goeswith",
#     "iobj",
#     "list",
#     "mark",
#     "nmod",
#     "nn",
#     "nsubj",
#     "nummod",
#     "number",
#     "obj",
#     "obl",
#     "orphan",
#     "partmod",
#     "parataxis",
#     "pcomp",
#     "pobj",
#     "possessive",
#     "punct",
#     "prep",
#     "reparandum",
#     "root",
#     "vocative",
#     "xcomp"
# ]
#
# DEP_REL += [
#             "dep",
#             "aux",
#             "auxpass",
#             "cop",
#             "conj",
#             "cc",
#             "arg",
#             "subj",
#             "nsubj",
#             "nsubjpass",
#             "csubj",
#             "comp",
#             "obj",
#             "dobj",
#             "iobj",
#             "pobj",
#             "attr",
#             "ccomp",
#             "xcomp",
#             "compl",
#             "mark",
#             "rel",
#             "acomp",
#             "agent",
#             "ref",
#             "expl",
#             "mod",
#             "advcl",
#             "purpcl",
#             "tmod",
#             "rcmod",
#             "amod",
#             "infmod",
#             "partmod",
#             "num",
#             "number",
#             "appos",
#             "nn",
#             "abbrev",
#             "advmod",
#             "neg",
#             "poss",
#             "possessive",
#             "prt",
#             "det",
#             "prep",
#             "sdep",
#             "xsubj"
# ]
#
#
# DEP_REL = list(set(DEP_REL))
