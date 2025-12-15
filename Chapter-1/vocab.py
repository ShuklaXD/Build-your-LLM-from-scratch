class VocabGenerator:
    def __init__(self, words):
        self.words = words

    def build_vocab(self):
        unique_words = sorted(set(self.words))
        str_to_int = {word: idx for idx, word in enumerate(unique_words)}
        return str_to_int