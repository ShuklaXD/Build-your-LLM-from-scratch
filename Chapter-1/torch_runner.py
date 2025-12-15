import torch
import tiktoken
from dataset import GPTDatasetV1
from torch.utils.data import DataLoader
    
max_length = 4
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Inputs:\n", inputs)
    print("Inputs shape:", inputs.shape)

    # Token Embeddings
    token_embeddings = token_embedding_layer(inputs)
    print("\nToken Embedding Shape:\n", token_embeddings.shape)

    # Positional Embeddings
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print("Positional Embedding Shape:\n", pos_embeddings.shape)

    # Adding token and positional embeddings
    combined_embeddings = token_embeddings + pos_embeddings
    print("Combined Embedding Shape:\n", combined_embeddings.shape)