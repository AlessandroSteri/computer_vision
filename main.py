import os
import argparse
from gte.model import GroundedTextualEntailmentModel
from gte.utils.log import id_gen
from gte.utils.dic import index_map
from gte.info import MAX_LEN_P, MAX_LEN_H, SHUFFLED_DIR
from gte.preprocessing.dataset import datasets_to_word_set, words_to_dictionary, generate_shuffled_datasets
from gte.images import Image2vec


def main(options, ID):
    if not os.path.exists(SHUFFLED_DIR): generate_shuffled_datasets()

    words, labels, lens_p, lens_h = datasets_to_word_set()
    # print("Words: ", len(words))
    # print("Labels: ", labels)

    embeddings, word2id, id2word = words_to_dictionary(words, options.embedding_name, options.embedding_size)
    label2id, id2label = index_map(list(labels))

    # image2vec = Image2vec(has_model=True)
    # image2vec.compute_all_feats_and_store()

    # Model as context manager
    with GroundedTextualEntailmentModel(options, ID, embeddings, word2id, id2word, label2id, id2label) as gte_model:
        gte_model.fit()


if __name__ == '__main__':
    # ### CLI args ### #
    # {{{
    cmdLineParser = argparse.ArgumentParser()
    # cmdLineParser.add_argument("epoch", default=1, type=int, help="Number of full training-set iterations.")
    cmdLineParser.add_argument('--epoch', action="store", dest="epoch", default=1, type=int, help="Number of epoch.")
    # cmdLineParser.add_argument("batch_size", default=8, type=int, help="Number of samples per batch.")
    cmdLineParser.add_argument('--batch_size', action="store", dest="batch_size", default=32, type=int, help="Number of samples per batch.")
    # cmdLineParser.add_argument("embedding_name", type=str, help="Embedding vector to use.")
    cmdLineParser.add_argument('--embedding_name', action="store", dest="embedding_name", default="glove840", type=str, help="Embedding vector to use.")
    # cmdLineParser.add_argument("embedding_size", type=int, help="Dimension of the embedding vector.")
    cmdLineParser.add_argument('--embedding_size', action="store", dest="embedding_size", default=300, type=int, help="Dimension of the embedding vector.")
    # cmdLineParser.add_argument("learning_rate", default=0.01, type=float, help="Base learning rate.")
    cmdLineParser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.001, type=float, help="Learning rate.")
    # cmdLineParser.add_argument("step_check",  default=50, type=int, help="Every how many iteration check accuracy over dev sets.")
    cmdLineParser.add_argument('--step_check', action="store", dest="step_check", default=200, type=int, help="Every how many iteration check accuracy over dev sets.")
    # cmdLineParser.add_argument("hidden_size", type=int, help="Length of hidden layer.")
    cmdLineParser.add_argument('--hidden_size', action="store", dest="hidden_size", default=256, type=int, help="Length of hidden layer.")
    cmdLineParser.add_argument('--max_len_p', action="store", dest="max_len_p", default=MAX_LEN_P, type=int, help="Max lenght for premises.")
    cmdLineParser.add_argument('--max_len_h', action="store", dest="max_len_h", default=MAX_LEN_H, type=int, help="Max lenght for hypothesis.")
    cmdLineParser.add_argument('--bilstm_layer', action="store", dest="bilstm_layer", default=2, type=int, help="Num layer for context bilstm.")
    # cmdLineParser.add_argument('--match_layer', action="store", dest="match_layer", default=1, type=int, help="Num layer for matching.")
    # FLAGS
    cmdLineParser.add_argument("--gpu", dest="use_gpu", action='store_true', help="Enable gpu accelaration.")
    cmdLineParser.add_argument("--dev_env", dest="dev_env", action='store_true', help="Development Environment: Use always the same seed, experiments are repeatable.")
    cmdLineParser.add_argument("--trainable", dest="trainable", action='store_true', help="Makes trainable the pre-trained embeddigs.")
    cmdLineParser.add_argument("--with_matching", dest="with_matching", action='store_true', help="Makes use of bilateral matching.")
    cmdLineParser.add_argument("--with_img", dest="with_img", action='store_true', help="Makes use of bilateral matching.")
    cmdLineParser.add_argument("--with_img2", dest="with_img2", action='store_true', help="Makes use of bilateral matching.")
    cmdLineParser.add_argument("--dropout", dest="dropout", action='store_true', help="Applies dropout at the latent representation.")
    cmdLineParser.add_argument("--attentive", dest="attentive", action='store_true', help="Applies cross multihead attention to PI vs PI and HI vs HI.")
    cmdLineParser.add_argument("--attentive_swap", dest="attentive_swap", action='store_true', help="Applies cross multihead attention to PI vs HI and HI vs PI.")
    options = cmdLineParser.parse_args()
    # }}}

    # ### Unique Run ID with Hyperparameters Info ### #
    # {{{
    env = ''
    if options.dev_env: env = '[DEV]'

    model_info = ' B[{}]_Emb[{}]_lr[{}]_Hid[{}]_Lp[{}]_Lh[{}]_bilstm[{}]'.format(options.batch_size,
                                                                            options.embedding_name,
                                                                            options.learning_rate,
                                                                            options.hidden_size,
                                                                            options.max_len_p,
                                                                            options.max_len_h,
                                                                            options.bilstm_layer)
    if options.trainable: model_info += 'T.'.format()
    if options.with_matching: model_info += 'M.'.format()
    if options.with_img2 or options.with_img: model_info += 'I.'.format()
    if options.dropout: model_info += 'D.'.format()
    if options.attentive: model_info += 'A.'.format()
    if options.attentive_swap: model_info += 'Asw'.format()
    exe_id = id_gen() # sort-of unique and monotonic id for tensorboard and logging
    ID = exe_id + env + model_info
    # }}}

    # '_b' + str(options.batch_size) + '_e' + str(options.epoch) + '_l' + str(options.learning_rate) + '_c' + str(options.step_check) + '_model' + '--{}]'.format(

    # ### MAIN ### #
    # {{{
    main(options, ID)
    # }}}
