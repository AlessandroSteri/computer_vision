from gte.preprocessing.dataset import datasets_to_word_set, words_to_dictionary

def main():
    words, labels, lens_p, lens_h = datasets_to_word_set()
    print("Words: ", len(words))
    print("Labels: ", labels)

    embedding_name = 'glove'
    embedding_size = 50
    embedding, word_to_index, index_to_word = words_to_dictionary(words, embedding_name, embedding_size)
    import ipdb; ipdb.set_trace()  # TODO BREAKPOINT

if __name__ == '__main__':
    main()
