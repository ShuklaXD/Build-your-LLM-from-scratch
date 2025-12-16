import torch
import tiktoken
from chapter_1.dataset import GPTDatasetV1
from torch.utils.data import DataLoader

max_length = 4
vocab_size = 50257
output_dim = 15
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

class SimplifiedSelfAttentionDemo:
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
            # Self-Attention Demo
            # Taking a sample input: "Hello, do you like tea?"
            input = "Hello, do you like tea?"
            tokenizer = tiktoken.get_encoding("gpt2")
            input_ids = tokenizer.encode(input)
            print("\nInput IDs:", input_ids)
            input_tensor = torch.tensor(input_ids)
            print("\nInput Tensor:\n", input_tensor)
            input_embeddings = token_embedding_layer(input_tensor)
            self.attend_to_input(input_embeddings)
            # Calculating attention scores:
            # tensor([-15.2383, 238.6343, -13.1432, -11.4588,  -4.1859,   2.2816,   6.0516],
            #     grad_fn=<CopySlices>)

            # Attention weights (naive softmax): tensor([0., 1., 0., 0., 0., 0., 0.], grad_fn=<SoftmaxBackward0>)
            # The softmax was not uniform when we dealt with 256 dimensions.

            # Book example uses custom embeddings for clarity
            input_embeddings = torch.tensor(
                [[0.43, 0.15, 0.89], # Your
                [0.55, 0.87, 0.66], # journey
                [0.57, 0.85, 0.64], # starts
                [0.22, 0.58, 0.33], # with
                [0.77, 0.25, 0.10], # one
                [0.05, 0.80, 0.55]] # step
                )
            self.attend_to_input(input_embeddings)
            
    def attend_to_input(self, input_embeddings):
        attn_weights = torch.empty(input_embeddings.shape[0], input_embeddings.shape[0])  # To store attention scores
        print("\nInput Embeddings Shape:\n", input_embeddings.shape)

        # Calculating attention scores for the second token in the query "the comma"
        print("\nCalculating attention scores:")
        attn_weights = torch.matmul(input_embeddings, input_embeddings.T)
        # for i, x_i in enumerate(input_embeddings):
        #     for j, x_j in enumerate(input_embeddings):
        #         attn_weights[i, j] += torch.dot(x_i, x_j)
        #     # torch.softmax(attn_weights[i], dim=0)
        print(attn_weights)
        # By setting dim=-1, we are instructing the softmax function to apply the normalization along the last dimension of the attn_scores tensor.
        attn_weights = torch.softmax(attn_weights, dim=-1)
        print("\nAttention weights (softmax):", attn_weights)
        print("\nSum (softmax):", attn_weights.sum(dim=-1))
        return attn_weights

    def attend_to_input_single_query(self, input_embeddings, query_index=1):
        query = input_embeddings[query_index]  # Taking the embedding of the second token as query
        attn_weights_2 = torch.empty(input_embeddings.shape[0])  # To store attention scores
        for i, x_i in enumerate(input_embeddings):
            attn_weights_2[i] = torch.dot(x_i, query)
        print(attn_weights_2)

        attn_weights_softmax = torch.softmax(attn_weights_2, dim=0)
        print("\nAttention weights (softmax):", attn_weights_softmax)
        print("\nSum (softmax):", attn_weights_softmax.sum())
        return attn_weights_softmax