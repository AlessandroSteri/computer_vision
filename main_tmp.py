from gte.preprocessing.dataset import datasets_to_word_set, words_to_dictionary
from gte.preprocessing.batch import generate_batch
from gte.utils.dic import index_map
from gte.info import DEV_DATA, MAX_LEN_P, MAX_LEN_H


def main():
    words, labels, lens_p, lens_h = datasets_to_word_set()
    print("Words: ", len(words))
    print("Labels: ", labels)

    embedding_name = 'glove'
    embedding_size = 50
    embedding, word2id, id2word = words_to_dictionary(words, embedding_name, embedding_size)
    label2id, id2label = index_map(list(labels))
    batch_size = 32
    for batch in generate_batch(DEV_DATA, batch_size, word2id, label2id, max_len_p=MAX_LEN_P, max_len_h=MAX_LEN_H):
        print(batch.H)
        import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
        continue


if __name__ == '__main__':
    main()
