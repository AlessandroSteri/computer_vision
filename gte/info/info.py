TB_DIR = './log/tensorboard'
HP_LOG = './log/log.txt'
UNK = "<UNK>"

# NON SHUFFLED, FOLLOWS PATTERN
X_TRAIN_DATA = './DATA/vsnli/VSNLI_1.0_train.tsv'
X_DEV_DATA = './DATA/vsnli/VSNLI_1.0_dev.tsv'
X_TEST_DATA = './DATA/vsnli/VSNLI_1.0_test.tsv'
X_TEST_DATA_HARD = './DATA/vsnli/VSNLI_1.0_test_hard.tsv'

SHUFFLED_DIR = './DATA/vsnli/SHUFFLED'

TRAIN_DATA = './DATA/vsnli/SHUFFLED/VSNLI_1.0_train.tsv'
DEV_DATA = './DATA/vsnli/SHUFFLED/VSNLI_1.0_dev.tsv'
TEST_DATA = './DATA/vsnli/SHUFFLED/VSNLI_1.0_test.tsv'
TEST_DATA_HARD = './DATA/vsnli/SHUFFLED/VSNLI_1.0_test_hard.tsv'
IMG_DATA = './DATA/flickr30k_images'
IMG_FEATS = './DATA/flickr30k_images/features'

BEST_F1 = './log/f1.txt'

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

DEP_REL = [
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp"
]