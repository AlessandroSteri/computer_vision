from collections import namedtuple
import numpy as np

def dict_to_named_tuple(dictionary):
    for k, v in dictionary.items():
                if isinstance(v, dict):
                    dictionary[k] = dict_to_named_tuple(v)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

# given a list return a dictionary (and inv dict) elem -> index  (index -> elem)
# if unk is not None appends to dictionary as last index unk
def index_map(list_of_elem, unk=None, offset=0):
    dic = {elem: idx+offset for idx, elem in enumerate(list_of_elem)}
    inv_dic = {idx+offset: elem for idx, elem in enumerate(list_of_elem)}
    if unk is not None:
        assert len(list_of_elem) + offset not in inv_dic
        assert len(list_of_elem) + offset not in dic
        dic[unk] = len(list_of_elem) + offset
        inv_dic[len(list_of_elem) + offset] = unk
        #TODO: return len if unk none and len+1 if unk not none so to have coherent num of classes
    return dic, inv_dic

def invert_dictionary(dictionary):
    return dict(zip(dictionary.values(), dictionary.keys()))

def one_hot_encode(index, num_elements):
    one_hot_vector = [0] * num_elements #np.zeros(num_elements)
    one_hot_vector[index] = 1
    return one_hot_vector

def one_hot_decode(one_hot_vector, num_elements):
    return one_hot_vector.index(1)

def one_hot_encoding_embeddings(size):
    return np.identity(size, dtype=int)

def test_one_hot(index, num_elements):
    one_hot_vector = one_hot_encode(index, num_elements)
    index2 = one_hot_decode(one_hot_vector, num_elements)
    print("one_hot_vector: {}".format(one_hot_vector))
    print("index: {}".format(index2))
    assert index == index2

def dic_lookup(dictionary, key, out_of_dict_key):
    assert out_of_dict_key in dictionary
    if key in dictionary:
        return dictionary[key]
    else:
        return dictionary[out_of_dict_key]


def dic_lookup_case_sensitive(dictionary, key, out_of_dict_key):
    val = dic_lookup(dictionary, key, out_of_dict_key)
    if val == out_of_dict_key:
        val = dic_lookup(dictionary, key.lower(), out_of_dict_key)
    return val

