import numpy as np
import random

from embedding import no_stop_words_vocab, build_index_map, rule_zh, bow_whole, zh_stop_words

from stanza.models.common.doc import Document, Sentence

from prepare.clean import load_doc_list

random.seed(2020)


def build_word_vocab_random_replace_num_with_entity_new(core_list):
    ent_set = set()
    word_set = set()
    for doc in core_list:
        for sentence in doc.sentences:
            for ent in sentence.ents:
                ent_set.add(ent.text)
            for word in sentence.words:
                if word.text in ent_set:
                    continue
                word_set.add(word.text)

    whole_vocab = {}
    for doc in core_list:
        doc_vocab = {}
        for sentence in doc.sentences:
            sentence_vocab = {}

            ent_id = set()
            for ent in sentence.ents:
                for word in ent.words:
                    ent_id.add(word.id)

            wds = []
            for word in sentence.words:
                pos = word.pos
                if pos == 'PUNCT':
                    continue
                wds.append(word)


            choices = set(random.sample(wds, min(len(ent_id), len(wds))))

            for i, word in enumerate(sentence.words):
                word_text = word.text
                if word in choices:
                    word_text = random.sample(word_set, 1)[0]

                sentence_vocab[word_text] = sentence_vocab.setdefault(word_text, 0) + 1
                doc_vocab[word_text] = doc_vocab.setdefault(word_text, 0) + 1
                whole_vocab[word_text] = whole_vocab.setdefault(word_text, 0) + 1
            sentence.word_vocab = sentence_vocab
        doc.word_vocab = doc_vocab

    return whole_vocab



def build_word_vocab_random_replace_num_with_entity(core_list):
    ent_set = set()
    word_set = set()
    for doc in core_list:
        for sentence in doc.sentences:
            for ent in sentence.ents:
                ent_set.add(ent.text)
            for word in sentence.words:
                if word.text in ent_set:
                    continue
                word_set.add(word.text)

    whole_vocab = {}
    for doc in core_list:
        doc_vocab = {}
        for sentence in doc.sentences:
            sentence_vocab = {}

            ent_id = set()
            for ent in sentence.ents:
                for word in ent.words:
                    ent_id.add(word.id)
            choices = set(random.sample(sentence.words, len(ent_id)))

            for i, word in enumerate(sentence.words):
                word_text = word.text
                if word in choices:
                    word_text = random.sample(word_set, 1)[0]

                sentence_vocab[word_text] = sentence_vocab.setdefault(word_text, 0) + 1
                doc_vocab[word_text] = doc_vocab.setdefault(word_text, 0) + 1
                whole_vocab[word_text] = whole_vocab.setdefault(word_text, 0) + 1
            sentence.word_vocab = sentence_vocab
        doc.word_vocab = doc_vocab

    return whole_vocab


def build_word_vocab_no_ent_two(core_list):
    ent_set = set()
    word_set = set()
    for doc in core_list:
        for sentence in doc.sentences:
            for ent in sentence.ents:
                ent_set.add(ent.text)
            for word in sentence.words:
                if word.text in ent_set:
                    continue
                word_set.add(word.text)

    whole_vocab = {}
    for doc in core_list:
        doc_vocab = {}
        for sentence in doc.sentences:
            sentence_vocab = {}

            ent_id = set()
            for ent in sentence.ents:
                for word in ent.words:
                    ent_id.add(word.id)

            for i, word in enumerate(sentence.words):
                word_text = word.text
                if word_text in ent_set or i + 1 in ent_id:
                    word_text = random.sample(word_set, 1)[0]

                sentence_vocab[word_text] = sentence_vocab.setdefault(word_text, 0) + 1
                doc_vocab[word_text] = doc_vocab.setdefault(word_text, 0) + 1
                whole_vocab[word_text] = whole_vocab.setdefault(word_text, 0) + 1
            sentence.word_vocab = sentence_vocab
        doc.word_vocab = doc_vocab

    return whole_vocab


def build_word_vocab_no_ent(core_list):
    ent_set = set()
    word_set = set()
    for doc in core_list:
        for sentence in doc.sentences:
            for ent in sentence.ents:
                for word in ent.words:
                    ent_set.add(word.text)
            for word in sentence.words:
                if word.text in ent_set:
                    continue
                word_set.add(word.text)

    whole_vocab = {}
    for doc in core_list:
        doc_vocab = {}
        for sentence in doc.sentences:
            sentence_vocab = {}

            for i, word in enumerate(sentence.words):
                word_text = word.text
                if word_text in ent_set:
                    word_text = random.sample(word_set, 1)[0]

                sentence_vocab[word_text] = sentence_vocab.setdefault(word_text, 0) + 1
                doc_vocab[word_text] = doc_vocab.setdefault(word_text, 0) + 1
                whole_vocab[word_text] = whole_vocab.setdefault(word_text, 0) + 1
            sentence.word_vocab = sentence_vocab
        doc.word_vocab = doc_vocab

    return whole_vocab


vocab_path = ""

embed_list = load_doc_list(vocab_path)

# whole_vocab = build_word_vocab_no_ent(embed_list)
# whole_vocab = build_word_vocab_no_ent_two(embed_list)
whole_vocab = build_word_vocab_random_replace_num_with_entity(embed_list)
no_stop_words_vocab = no_stop_words_vocab(whole_vocab, zh_stop_words, rule_zh)

index_map = build_index_map(no_stop_words_vocab)

embedding = bow_whole(embed_list, no_stop_words_vocab, index_map, 2000)
# np.savetxt("data/ablation_no_ent/no_ent_embedding", X=embedding, fmt='%.6f')
# np.savetxt("data/ablation_no_ent_two/no_ent_embedding", X=embedding, fmt='%.6f')
np.savetxt("data/ablation_no_ent_two/random_replace_num_with_entity", X=embedding, fmt='%.6f')