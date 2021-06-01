import numpy as np


def load_set(filename):
    r = []
    with open(filename,encoding='utf-8') as file:
        for line in file.readlines():
            a = [str(x) for x in line.strip("\n").strip("[]").split(",") if x]
            r.append(a)
        file.close()
    return r


def get_vocab(text_list):
    vocabulary = {}
    text_dic = []
    for word_list in text_list:
        dic = {}
        for word in word_list:
            if word in dic:
                dic[word] = dic[word] + 1
            else:
                dic[word] = 1

            if word in vocabulary:
                vocabulary[word] = vocabulary[word] + 1
            else:
                vocabulary[word] = 1
        text_dic.append(dic)
    return text_dic, vocabulary


def bow(text_dic, vocabulary: dict, dim=None):
    if not dim:
        dim = len(vocabulary)

    x = np.zeros((len(text_dic), dim), dtype=float)
    sorted_list = sorted(vocabulary.items(),key=lambda d:d[1], reverse=True)
    index_dic = {}
    for i,k in enumerate(sorted_list):
        index_dic[k[0]] = i
    for i, dic in enumerate(text_dic):
        for word in dic:
            if word in vocabulary:
                if index_dic[word] < dim:
                    x[i][index_dic[word]] = dic[word]
    return x


text_set = load_set('data/thucnews')
text_dic, vocab = get_vocab(text_set)
x = bow(text_dic, vocab, dim=2000)