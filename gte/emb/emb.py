import os
import numpy as np
from gte.utils.dic import index_map
from gte.utils.ser import serialize, deserialize, exists_serialized_obj
from gte.utils.log import LogTime
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from sys import platform


def retrieve_embeddings(embedding_name, embedding_size, words_domain):
    lite_emb_caching_dir = os.path.join('log/embeddings/lite', embedding_name, str(embedding_size))
    lite_emb_name = 'vec'
    emb_dict_name = 'dic'
    emb_inv_dict_name = 'inv_dic'
    lite_embedding, word_to_index, index_to_word = None, None, None
    if not exists_serialized_obj(lite_emb_name, base_dir=lite_emb_caching_dir) or not exists_serialized_obj(emb_dict_name, base_dir=lite_emb_caching_dir) or not exists_serialized_obj(emb_inv_dict_name, base_dir=lite_emb_caching_dir):
        with LogTime('Selecting Embeddings from scratch'):
            words, embeddings = get_embeddings(embedding_name, embedding_size)

            lite_embedding, word_to_index, index_to_word = select_embeddings_of_interest(words_domain, words, embeddings)
            serialize(lite_embedding, lite_emb_name, base_dir=lite_emb_caching_dir)
            serialize(word_to_index, emb_dict_name, base_dir=lite_emb_caching_dir)
            serialize(index_to_word, emb_inv_dict_name, base_dir=lite_emb_caching_dir)
    else:
        with LogTime('Loading pickled selected embeddings'):
            lite_embedding = deserialize(lite_emb_name, base_dir=lite_emb_caching_dir)
            word_to_index = deserialize(emb_dict_name, base_dir=lite_emb_caching_dir)
            index_to_word = deserialize(emb_inv_dict_name, base_dir=lite_emb_caching_dir)


    out_vocab_rate = (len(words_domain) - len(lite_embedding)) / len(words_domain)
    print("Emb: ", len(lite_embedding), 'Words: ', len(words_domain))
    # print("Out of vocab rate: ", out_vocab_rate)
    return lite_embedding, word_to_index, index_to_word

def get_embeddings(name, size):
    words, vectors = [], []
    if name == 'glove':
        words, vectors = get_glove_embeddings(size)
    if name == 'glove42':
        words, vectors = get_glove_embeddings(size, b42=True)
    if name == 'glove840':
        words, vectors = get_glove_embeddings840(size)
    if name == 'google':
        words, vectors = get_google_embeddings(size)
    return words, vectors

def get_glove_embeddings840(size):
    assert size == 300

    path = 'glove/glove.840B.300d.txt'
    words = []
    vectors = []
    model = {}
    with open(path,'r', encoding='utf8') as f:
        for line in f:
            line_split = line.split(' ')
            word = line_split[0]
            embedding = np.asarray(line_split[1:], dtype='float32')
            model[word] = embedding
            words.append(word)
            vectors.append(embedding)
    return words, vectors


def get_glove_embeddings(size, b42=False):
    path = ''
    if size == 50:
        path = 'glove/glove.6B.50d.txt'
    if size == 100:
        path = 'glove/glove.6B.100d.txt'
    if size == 200:
        path = 'glove/glove.6B.200d.txt'
    if size == 300 and not b42:
        path = 'glove/glove.6B.300d.txt'
    if size == 300 and b42:
        path = 'glove/glove.42B.300d.txt'
    if platform.startswith('linux'):
        glove2word2vec(glove_input_file=path, word2vec_output_file="gensim_glove_vectors.txt")
        from gensim.models.keyedvectors import KeyedVectors
        glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
        vectors = np.zeros((len(glove_model.wv.vocab), size))
        words = np.chararray((len(glove_model.wv.vocab)), unicode=True)
        for i in range(len(glove_model.wv.vocab)):
            embedding_vector = glove_model.wv[glove_model.wv.index2word[i]]
            if embedding_vector is not None:
                vectors[i] = embedding_vector
                words[i] = glove_model.wv.index2word[i]
    elif platform.startswith('darwin'):
        glove = np.loadtxt(path, dtype='str', comments=None)
        words = glove[:, 0]
        vectors = glove[:, 1:].astype('float')    
    return words, vectors


def select_embeddings_of_interest(words_domain, words, embeddings):
    custom_embedding = []
    custom_dic = dict()
    custom_inv_dic = dict()
    index = 0
    emb_dic, emb_inv_dic = index_map(words)

    # words for that appear either in train, dev or test set
    # for which an embedding is available
    words_to_embed = set(words).intersection(set(words_domain))

    for w in words_to_embed:
        emb_idx = emb_dic[w]
        vector = embeddings[emb_idx]
        custom_embedding.append(vector)
        custom_dic[w] = index
        custom_inv_dic[index] = w
        assert custom_embedding[index].all() == embeddings[emb_dic[w]].all()
        assert custom_embedding[custom_dic[w]].all() == embeddings[emb_dic[w]].all()
        index += 1
    # avg vectors to define unk vector
    custom_embedding.append(np.mean(custom_embedding, axis=0))
    custom_dic["<UNK>"] = index
    custom_inv_dic[index] = "<UNK>"
    np_custom_emb = np.asarray(custom_embedding)

    # assert type(custom_embedding) == type(embeddings)
    return np.asarray(custom_embedding), custom_dic, custom_inv_dic

def get_google_embeddings(size):
    assert size == 300
    model = gensim.models.KeyedVectors.load_word2vec_format('google/GoogleNews-vectors-negative300.bin', binary=True)
    words = model.index2word
    vectors = model.vectors
    return words, vectors
