import argparse
import os
import pickle
import random

from prepare.clean import load_doc_list


def add_sentence(doc, sent):
    add_token_num = len(sent.tokens)
    add_word_num = len(sent.words)
    add_text = sent.text

    sen = doc.sentences
    sen.append(sent)

    doc.sentences = sen
    doc.num_tokens = doc.num_tokens + add_token_num
    doc.num_words = doc.num_words + add_word_num
    doc.text = doc.text + add_text
    # doc.build_ents()


def add_sentences(doc, sents):
    for sent in sents:
        add_sentence(doc, sent)
    doc.build_ents()


def expend_data(core_list, s_list, s_o_list, s_o_n_list, expected_len, alpha=0.5, beta=0.75):
    # s = self, o = other, n = noisy

    n = len(core_list)
    noisy_index = []
    index_list = [i for i in range(n)]
    random.shuffle(index_list)

    s_expend = [[] for _ in range(n)]
    s_o_expend = [[] for _ in range(n)]

    count = 0

    sen_2_entity = {}

    for doc in core_list:
        for sen in doc.sentences:
            sen_2_entity[sen] = set([ent.text for ent in sen.ents])

    for i in index_list:
        count += 1
        if count % 200 == 0:
            print("do 200")
        add_sent = []
        i_doc = core_list[i]
        len_s = len(i_doc.sentences)
        if len_s >= expected_len:
            noisy_index.append(i)
            need_len = int(len_s * (1 - beta))
        else:
            # 增加自己的信息
            m = len_s
            while m < expected_len * beta:
                r = random.randint(0, len_s - 1)
                s_expend[i].append(i_doc.sentences[r])
                s_o_expend[i].append(i_doc.sentences[r])
                m += 1

            need_len = expected_len - m

        for i_sen in i_doc.sentences:

            for j in index_list:
                if i == j:
                    continue
                j_doc = core_list[j]

                for j_sen in j_doc.sentences:
                    i_ent_set = sen_2_entity[i_sen]
                    j_ent_set = sen_2_entity[j_sen]

                    len_inter = len(i_ent_set.intersection(j_ent_set))
                    len_union = len(i_ent_set.union(j_ent_set))
                    if len_union == 0:
                        continue

                    if len_inter / len_union >= alpha:
                        add_sent.append((j_sen, len_inter / len_union))

        add_sent = [s for s, _ in sorted(add_sent, key=lambda d: d[1], reverse=True)[:need_len]]
        s_o_expend[i].extend(add_sent)

    noisy_set = set(noisy_index)
    for i in range(n):
        add_sentences(s_list[i], s_expend[i])
        add_sentences(s_o_n_list[i], s_o_expend[i])
        if i in noisy_set:
            continue
        add_sentences(s_o_list[i], s_o_expend[i])

    no_n_list = []
    n_list = []
    noisy_index = sorted(noisy_index, key=lambda d: d)
    for i in noisy_index:
        no_n_list.append(core_list[i])
        n_list.append(s_o_n_list[i])

    return no_n_list, n_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="process",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--process_path", type=str, default="")
    parser.add_argument("--expected_len", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--s_path", type=str, default="")
    parser.add_argument("--s_o_path", type=str, default="")
    args = parser.parse_args()

    process_path = args.process_path
    expected_len = args.expected_len
    alpha = args.alpha
    beta = args.beta
    s_path = args.s_path
    s_o_path = args.s_o_path

    core_list = load_doc_list(process_path)
    s_list = load_doc_list(process_path)
    s_o_list = load_doc_list(process_path)
    s_o_n_list = load_doc_list(process_path)
    no_n_list, n_list = expend_data(core_list, s_list, s_o_list, s_o_n_list, expected_len, alpha=alpha, beta=beta)

    with open(s_path, mode='wb') as file:
        pickle.dump(s_list, file)

    with open(s_o_path, mode='wb') as file:
        pickle.dump(s_o_list, file)