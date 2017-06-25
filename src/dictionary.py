import logging
import data_util

class Dict:
    TOKEN_PAD = '<PAD>'  # token for padding sentence
    TOKEN_UNK = '<UNK>'  # token for unknown words
    TOKEN_EOS = '<EOS>'  # token for End Of Sentence
    TOKEN_GO = '<GO>'  # token to start a sentence

    def __init__(self, name='unknown dict name'):

        self._name = name

        self._marks = [self.TOKEN_PAD, self.TOKEN_UNK, self.TOKEN_EOS, self.TOKEN_GO]

        self._words = {}

        self.token_to_id = dict()
        self.id_to_token = dict()

    def add_word(self, new_word):
        """
        add_word adds a single word to the dictionary corpus.
        It's when the corpus is complete that the dictionary can be build.

        :param new_word: a word to add to the dictionary
        """

        self._words[new_word] = self._words.get(new_word, 0) + 1

    def add_words(self, new_words):
        """
        add_words adds a list of words to the dictionary corpus.
        It's when the corpus is complete that the dictionary can be build.

        :param new_words: a list of words to add to the dictionary
        """

        for new_word in new_words:
            self._words[new_word] = self._words.get(new_word, 0) + 1

    def __str__(self):
        return "{Dict: name={}, size={}}".format(self._name, len(self.token_to_id))

    def create(self, max_vocab=None):
        """
        create creates the dictionary from all the words previously stored (in _words).
        A dictionary has 2 maps:
          - one for ids (to get the tokens),
          - one for the tokens (to get the matching ids).

        :param max_vocab: limit of the number of tokens in the dictionary (None by default)
        :return:
        """
        logging.info("Start creating dictionary '%s'", self._name)

        # Remove the special tokens from the corpus if there is any.
        for m in self._marks:
            if m in self._words:
                del self._words[m]
                logging.warning("%s appears in corpus", m)

            m = str.lower(m)
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

    def save(self, filename):
        """
        save stores the dictionary on a file with the format 'token_id token_value\n'.

        :param filename: name of the file which will store the dictionary
        """

        with open(filename, 'w') as dict_file:
            for idx, token in self.id_to_token.items():
                dict_file.write("{} {}\n".format(idx, token))

        logging.info("Dictionary '%s' has been saved in file '%s'", self._name, filename)

    def load(self, filename):
        """
        load loads the dictionary from a file.

        :param filename: name of the file containing the dictionary
        :return:
        """

        try:
            with open(filename) as dict_file:
                dict_data = dict_file.readlines()
        except FileNotFoundError:
            logging.warning("Load dictionary '%s' from file '%s' failed", self._name, filename)
            return

        dict_data = list(map(lambda x: x.split(), dict_data))

        self.token_to_id = dict(map(lambda x: (x[1], int(x[0])), dict_data))
        self.id_to_token = dict(map(lambda x: (int(x[0]), x[1]), dict_data))

        logging.info("Loaded dictionary '%s' with %s words.", self._name, len(self.token_to_id))

    def convert_tokens_to_ids(self, tokens):
        """
        convert_tokens_to_ids converts a list of words (aka token) into a list of ids.

        :param tokens: a list of tokens to convert
        """

        id_token_unknown = self.token_to_id[self.TOKEN_UNK]
        ids = []

        # for each tokens of the list, get the corresponding id, or else, the id of the unknown token.
        for token in tokens:
            ids.append(self.token_to_id.get(token, id_token_unknown))

        return ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')

    docid, sumid, = data_util.load_data("test", "data/test.txt", "data/test_dict.txt")
