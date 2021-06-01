import argparse
import math
import os
import re
import numpy as np

from nltk.corpus import stopwords

from stanza.models.common.doc import Document, Sentence

from prepare.clean import load_doc_list


def vocab_setter(self, word_vocab):
    self._word_vocab = word_vocab


Document.add_property('word_vocab', default={},
                      getter=lambda self: self._word_vocab,
                      setter=vocab_setter)
Sentence.add_property('word_vocab', default={},
                      getter=lambda self: self._word_vocab,
                      setter=vocab_setter)

en_stop_words = set(stopwords.words('english')).union({"'s", "'d", "'t", "'re", "'ve", "'ll"})
zh_stop_words = set()
with open("data/stop_words.txt", encoding="utf-8") as file:
    for word in file.readlines():
        zh_stop_words.add(word.strip())


def rule_zh(w):
    return re.match("^[\\u4e00-\\u9fa5_a-zA-Z0-9]+$", w)


def rule_en(w):
    return re.match("^\w+$", w)


def build_word_vocab(core_list):
    whole_vocab = {}
    for doc in core_list:
        doc_vocab = {}
        for sentence in doc.sentences:
            sentence_vocab = {}
            masked = np.random.randint(100, size=len(sentence.words))
            for i, word in enumerate(sentence.words):
                word_text = word.text

                sentence_vocab[word_text] = sentence_vocab.setdefault(word_text, 0) + 1
                doc_vocab[word_text] = doc_vocab.setdefault(word_text, 0) + 1
                whole_vocab[word_text] = whole_vocab.setdefault(word_text, 0) + 1
            sentence.word_vocab = sentence_vocab
        doc.word_vocab = doc_vocab
    return whole_vocab


def build_word_idf(core_list, whole_vocab):
    len_d = len(core_list)
    word_idf = {}
    for word in whole_vocab:
        idf_count = 0
        for doc in core_list:
            if word in doc.word_vocab:
                idf_count += 1
        idf = math.log((len_d + 1.0) / (idf_count + 1.0)) + 1.0
        word_idf[word] = idf
    return word_idf


def build_index_map(word_vocab):
    return {k: i for i, k in enumerate(word_vocab.keys())}


def no_stop_words_vocab(word_vocab, stop_words, rule):
    return {k: v for k, v in word_vocab.items() if k.lower() not in stop_words and rule(k)}


def bow(word_vocab, index_dic, doc_vocab):
    len_v = len(word_vocab)
    x = np.zeros(len_v, dtype=np.float32)
    for word in doc_vocab:
        if word not in word_vocab:
            continue
        index = index_dic[word]
        x[index] = doc_vocab[word]
    return x


def bow_whole(core_list, word_vocab, index_dic, dim=None):
    len_d = len(core_list)
    len_v = len(word_vocab)
    matrix = np.zeros((len_d, len_v), dtype=np.float32)
    for i, doc in enumerate(core_list):
        matrix[i] = bow(word_vocab, index_dic, doc.word_vocab)

    if not dim:
        dim = len_v

    if len_v < dim:
        raise AttributeError("word less then dim")

    sorted_list = sorted(word_vocab.items(), key= lambda d: d[1], reverse=True)[:dim]
    print(sorted_list)
    index_list = [index_dic[w] for w, _ in sorted_list]
    return matrix[:, index_list]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument("--dim", type=int, default=2000)
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--lang", type=str)
    args = parser.parse_args()

    path = args.data_path
    vocab_path = args.vocab_path
    dim = args.dim
    save_path = args.save_path
    lang = args.lang

    if lang == 'zh':
        rule = rule_zh
        swords = zh_stop_words
    elif lang == 'en':
        rule = rule_en
        swords = en_stop_words
    else:
        raise AttributeError("no this lang")

    wait_embed_list = load_doc_list(path)

    whole_vocab = build_word_vocab(load_doc_list(vocab_path))
    no_stop_words_vocab = no_stop_words_vocab(whole_vocab, swords, rule)
    index_map = build_index_map(no_stop_words_vocab)

    build_word_vocab(wait_embed_list)
    embedding = bow_whole(wait_embed_list, no_stop_words_vocab, index_map, dim)
    np.savetxt(save_path, X=embedding, fmt='%.6f')