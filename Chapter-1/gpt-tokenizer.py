from importlib.metadata import version
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

def example_usage():
    text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
    )
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(integers)

    strings = tokenizer.decode(integers)
    print(strings)

def tokenize_verdict():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    enc_text = tokenizer.encode(
        raw_text, allowed_special={"<|endoftext|>"}
    )
    print("Total number of tokens:", len(enc_text))
    # sample testing
    print("First 20 tokens:", enc_text[:20])
    strings = tokenizer.decode(enc_text)
    print("First 500 characters of decoded text:")
    print(strings[:500])

    # encoding a sample from the middle
    enc_sample = enc_text[50:]
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print(f"x: {x}")
    print(f"y:    {y}")

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

if __name__ == "__main__":
    # example_usage()
    tokenize_verdict()