import logging

import os.path
import numpy as np
import dictionary

MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3


def load_dict(dict_path, max_vocab=None):
    logging.info("Try loading dict from %s", dict_path)
    try:
        dict_file = open(dict_path)
        dict_data = dict_file.readlines()
        dict_file.close()
    except:
        logging.info("Load dict %s failed, create later", dict_path)
        return None

    dict_data = list(map(lambda x: x.split(), dict_data))
    if max_vocab:
        dict_data = list(filter(lambda x: int(x[0]) < max_vocab, dict_data))
    tok2id = dict(map(lambda x: (x[1], int(x[0])), dict_data))
    id2tok = dict(map(lambda x: (int(x[0]), x[1]), dict_data))
    logging.info("Load dict %s with %s words.", dict_path, len(tok2id))
    return tok2id, id2tok


def create_dict(dict_path, corpus, max_vocab=None):
    logging.info("Create dict %s", dict_path)
    counter = {}
    for line in corpus:
        for word in line:
            try:
                counter[word] += 1
            except:
                counter[word] = 1

    for mark_t in MARKS:
        if mark_t in counter:
            del counter[mark_t]
            logging.warning("%s appears in corpus.", mark_t)

    counter = list(counter.items())
    counter.sort(key=lambda x: -x[1])
    words = list(map(lambda x: x[0], counter))
    words = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO] + words
    if max_vocab:
        words = words[:max_vocab]

    tok2id = dict()
    id2tok = dict()
    with open(dict_path, 'w') as dict_file:
        for idx, tok in enumerate(words):
            print(idx, tok, file=dict_file)
            tok2id[tok] = idx
            id2tok[idx] = tok

    logging.info("Create dict %s with %d words", dict_path, len(words))
    return tok2id, id2tok


def corpus_map2id(data, tok2id):
    ret = []
    unk = 0
    tot = 0
    for doc in data:
        tmp = []
        for word in doc:
            tot += 1
            try:
                tmp.append(tok2id[word])
            except:
                tmp.append(ID_UNK)
                unk += 1
        ret.append(tmp)
    return ret, (tot - unk) / tot


def sen_map2tok(sen, id2tok):
    return list(map(lambda x: id2tok[x], sen))


def load_data(data_name,
              data_filename,
              data_dict_path,
              max_data_vocab=None):
    """

    :param data_name:
    :param data_filename:
    :param data_dict_path:
    :param max_data_vocab:
    :return:
    """

    logging.info("Loading data from %s", data_filename)

    data_dict = dictionary.Dict(data_name)

    if os.path.isfile(data_dict_path):
        # The dictionary already exists so we can load it from the file

        logging.info("Loading dictionary from '%s'", data_dict_path)

        data_dict.load(data_dict_path)
    else:
        # The dictionary doesn't exist so we have to create it from the corpus

        logging.info("Creating dictionary")

        with open(data_filename) as data_file:
            for line in data_file:
                data_dict.add_words(line.split())

        # Build the dictionary from all the words of the corpus
        data_dict.create(max_data_vocab)

        # Save the dictionary to file
        data_dict.save(data_dict_path)

    data_ids = []
    with open(data_filename) as data_file:
        for line in data_file:
            data_ids.append(data_dict.convert_tokens_to_ids(line.split()))

    logging.info("Data ids list created")

    return data_ids, data_dict


def load_valid_data(data_filename,
                    data_dict):
    logging.info("Load validation document from %s", data_filename)

    data_ids = []
    with open(data_filename) as data_file:
        for line in data_file:
            data_ids.append(data_dict.convert_tokens_to_ids(line.split()))

    logging.info("Data ids list created")

    return data_ids


def corpus_preprocess(corpus):
    import re
    ret = []
    for line in corpus:
        x = re.sub('\\d', '#', line)
        ret.append(x)
    return ret


def sen_postprocess(sen):
    return sen


def load_test_data(doc_filename, doc_dict):
    logging.info("Load test document from %s", doc_filename)

    with open(doc_filename) as docfile:
        docs = docfile.readlines()
    docs = corpus_preprocess(docs)

    logging.info("Load %d testing documents", len(docs))
    docs = list(map(lambda x: x.split(), docs))

    docid, cover = corpus_map2id(docs, doc_dict[0])
    logging.info("Doc dict covers %.2f%% words", cover * 100)

    return docid


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')

    docid, doc_dict = load_data("document", "data/train.article.txt", "data/doc_dict.txt", 30000)

    sumid, sum_dict = load_data("summary", "data/train.title.txt", "data/sum_dict.txt", 30000)

    checkid = np.random.randint(len(docid))
    print(checkid)

    print(docid[checkid], sen_map2tok(docid[checkid], doc_dict[1]))
    print(sumid[checkid], sen_map2tok(sumid[checkid], sum_dict[1]))

    val_docid = load_valid_data("data/valid.article.filter.txt", doc_dict)

    val_sumid = load_valid_data("data/valid.title.filter.txt", sum_dict)

    checkid = np.random.randint(len(val_docid))
    print(checkid)

    print(val_docid[checkid], sen_map2tok(val_docid[checkid], doc_dict[1]))
    print(val_sumid[checkid], sen_map2tok(val_sumid[checkid], sum_dict[1]))

    test_docid = load_test_data("data/test.giga.txt", doc_dict)

    checkid = np.random.randint(len(test_docid))
    print(checkid)

    print(test_docid[checkid], sen_map2tok(test_docid[checkid], doc_dict[1]))
