import torch
import tiktoken
from .dataset import GPTDatasetV1
from torch.utils.data import DataLoader

max_length = 4
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

class TorchRunner:
    def __init__(self):
        pass
    
    @staticmethod
    def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        return dataloader

    def run(self):
        with open("the-verdict.txt", "r", encoding="utf-8") as f:
            raw_text = f.read()
            dataloader = self.create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=4, shuffle=False)
            data_iter = iter(dataloader)
            
            # following code is only for one batch demonstration
            inputs, targets = next(data_iter)
            print("Inputs:\n", inputs)
            # Till now we have tokens in the form of integers
            # Example output:
            #  tensor([[   40,   367,  2885,  1464],
            #          [ 1807,  3619,   402,   271],
            #          [10899,  2138,   257,  7026],
            #          [15632,   438,  2016,   257],
            #          [  922,  5891,  1576,   438],
            #          [  568,   340,   373,   645],
            #          [ 1049,  5975,   284,   502],
            #          [  284,  3285,   326,    11]])
            print("Inputs shape:", inputs.shape)
            # Output: torch.Size([8, 4])
            # 8 -> batch size
            # 4 -> context length (number of tokens in each batch)

            # Token Embeddings - embed tokens into vectors of 256 dimensions
            token_embeddings = token_embedding_layer(inputs)
            print("\nToken Embedding Shape:\n", token_embeddings.shape)
            # Output: torch.Size([8, 4, 256])
            # 8 -> batch size 
            # 4 -> context length (number of tokens in each batch)
            # 256 -> embedding dimension

            # Positional Embeddings
            context_length = 4 # maximum input length for the LLM
            pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
            # Output: torch.Size([4, 256])
            # 4 -> input size (context length)
            # 256 -> embedding dimension
            pos_embeddings = pos_embedding_layer(torch.arange(context_length))
            print("Positional Embedding Shape:\n", pos_embeddings.shape)

            # Adding token and positional embeddings
            combined_embeddings = token_embeddings + pos_embeddings
            print("Combined Embedding Shape:\n", combined_embeddings.shape)
