import tokenizer
import vocab
import re


if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Preprocessing the text into string tokens
    preprocessed = tokenizer.SimpleTokenizerV1.extract_tokens(None, raw_text)
    print(len(preprocessed))
    print(preprocessed[:20])

    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)

    print("Total number of words:", len(preprocessed))
    vocab_gen = vocab.VocabGenerator(preprocessed)
    vocab = vocab_gen.build_vocab()

    print("Vocabulary size:", len(vocab))
    tokenizer_v1 = tokenizer.SimpleTokenizerV1(vocab)

    sample_text = "Hello, do you like tea?"
    encoded = tokenizer_v1.encode(sample_text)
    decoded = tokenizer_v1.decode(encoded)

    print("Sample text:", sample_text)
    print("Encoded:", encoded)
    print("Decoded:", decoded)