import argparse
import json
import pickle
import re
import stanza

from stanza.models.common.doc import Document

def load_data_weibo(path: str):
    index = 0
    story_id = 0
    with open(path, mode='r', encoding='utf-8-sig', errors='ignore') as file:
        all_story = json.load(file)
        for story in all_story:
            document_list = [d for d in story['lsit'] if d['correlation'] == '完全相关']
            if len(document_list) < 100:
                continue
            print(len(document_list))

            for document in document_list:
                if document['correlation'] != "完全相关":
                    continue

                content = document['detail']
                content = re.sub("\\xad|\\u200b|\\x96|\\x95|\\x80|\\x9c|\\x9d|\\u200b|\\ue627|\\xa0", "", content)
                # content = re.sub("#[^#]+#|【[^【|】]+】", lambda m: m.group().strip("#|【|】") + "。", content)

                # yield index, story_id, content
                yield content
                index += 1
            story_id += 1


def load_stackoverflow(path:str):

    with open(path, mode='r', encoding='utf-8-sig', errors='ignore') as file:
        contents = file.readlines()

    for content in contents:
        content = content.strip("\n")
        yield content


def build(lang, data_loader, requirement):
    core_list = []
    core_nlp = stanza.Pipeline(lang, "stanza_model", processors=requirement, use_gpu=False)
    for raw_text in data_loader:
        core_doc = core_nlp(raw_text)
        core_list.append(core_doc)
    return core_list


def load_doc_list(path):
    with open(path, mode='rb') as file:
        return pickle.load(file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="process",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, default="weibo")
    parser.add_argument("--save_path", type=str, default="process")
    args = parser.parse_args()
    name = args.name
    save_path = args.save_path

    if name == "weibo":
        loader = load_data_weibo('data/weibo.json')
        data_list = build("zh", loader, "tokenize,pos,lemma,depparse,ner")
    elif name == "stackoverflow":
        loader = load_stackoverflow('data/stackoverflow.txt')
        data_list = build("en", loader, "tokenize")
    else:
        raise AttributeError("no this dataset")

    with open(save_path, mode='wb') as file:
        pickle.dump(data_list, file)
