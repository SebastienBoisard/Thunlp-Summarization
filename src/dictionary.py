import logging


class Dict:

    def __init__(self, name='unknown dict name'):

        self._name = name

        self._marks = ["<PAD>", "<UNK>", "<EOS>", "<GO>"]

        self._words = {}

        self.token_to_id = dict()
        self.id_to_token = dict()

    def add_word(self, new_word):
        self._words[new_word] = self._words.get(new_word, 0) + 1

    def add_words(self, new_words):
        for new_word in new_words:
            self._words[new_word] = self._words.get(new_word, 0) + 1

    def __str__(self):
        return "_words="+str(self._words)

    def create(self, max_vocab=None):
        logging.info("Start creating dictionary '%s'", self._name)

        for m in self._marks:
            if m in self._words:
                del self._words[m]
                logging.warning("%s appears in corpus", m)

        # Create a list of tuples (token, number of times the token appeared)
        counter = list(self._words.items())

        # Sort tokens from the most encountered to the least encountered
        counter.sort(key=lambda x: -x[1])

        # Extract tokens from the sorted list of tuples (token, number of times the token appeared)
        words = list(map(lambda x: x[0], counter))

        del counter

        # Add the special tokens at the beginning of the token list
        words = self._marks + words

        # Test if the dictionary has a maximum size
        if max_vocab:
            # Reduce the size of the dictionary
            words = words[:max_vocab]

        # Create 2 maps to get an id from a token, and a token from an id
        for idx, tok in enumerate(words):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

        del words

        logging.info("Dictionary '%s' has been created with %d tokens", self._name, len(self.token_to_id))

    def save(self, dict_filename):
        with open(dict_filename, 'w') as dict_file:
            for idx, token in self.id_to_token.items():
                dict_file.write("{} {}\n".format(idx, token))

        logging.info("Dictionary '%s' has been saved in file %s", self._name, dict_filename)

    def convert_tokens_to_ids(self, tokens):
        ids = []
        UNK_id = self.token_to_id['<UNK>']

        for token in tokens:
            ids.append(self.token_to_id.get(token, UNK_id))

        return ids
