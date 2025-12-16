class VocabGenerator:
    def __init__(self, words):
        self.words = words

    def build_vocab(self):
        unique_words = sorted(set(self.words))
        
        # Add special tokens towards the end of the vocabulary
        vocab = list(unique_words)
        vocab.extend(["<|endoftext|>", "<|unk|>"])

        str_to_int = {word: idx for idx, word in enumerate(vocab)}
        return str_to_int