import torch
import tiktoken
from chapter_1.dataset import GPTDatasetV1
from torch.utils.data import DataLoader

max_length = 4
vocab_size = 50257
output_dim = 15
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

class SelfAttentionDemo:
    def __init__(self):
        pass

    def run(self):
        # Book example uses custom embeddings for clarity
        input_embeddings = torch.tensor(
            [[0.43, 0.15, 0.89], # Your
            [0.55, 0.87, 0.66], # journey
            [0.57, 0.85, 0.64], # starts
            [0.22, 0.58, 0.33], # with
            [0.77, 0.25, 0.10], # one
            [0.05, 0.80, 0.55]] # step
            )
        self.attend_to_input_single_query(input_embeddings)

    def attend_to_input_single_query(self, input_embeddings, query_index=1):
        query_token = input_embeddings[query_index]  # Taking the embedding of the second token as query
        dimension_in = input_embeddings.shape[1] # Embedding dimension
        dimension_out = 2     # Reduced dimension for query, key, value

        torch.manual_seed(123)
        W_query = torch.nn.Parameter(torch.rand(dimension_in, dimension_out), requires_grad=False)
        W_key = torch.nn.Parameter(torch.rand(dimension_in, dimension_out), requires_grad=False)
        W_value = torch.nn.Parameter(torch.rand(dimension_in, dimension_out), requires_grad=False)

        # generate query, key, value vectors for the query token
        query_qt = query_token @ W_query
        key_qt = query_token @ W_key
        value_qt = query_token @ W_value
        print(query_qt)

        # generate key and value vectors for all tokens
        keys = input_embeddings @ W_key
        values = input_embeddings @ W_value
        print("\nKeys:\n", keys)
        print("\nValues:\n", values)
        print("\nKey for the query token:\n", key_qt)
        print("\nValue for the query token:\n", value_qt)
        # here key_qt == keys[query_index] and value_qt == values[query_index]

        # For single query token attention score calculation
        keys_qt = keys[query_index]
        attn_score_qt = query_qt.dot(keys_qt)
        print("\nAttention score for the query token:", attn_score_qt)

        # For all tokens attention score calculation
        attn_scores_qt = query_qt @ keys.T
        print("\nAttention scores for all tokens:", attn_scores_qt)

        # Softmax normalization for attention weights
        d_k = keys.shape[-1]
        attn_weights_qt = torch.softmax(attn_scores_qt / d_k**0.5, dim=-1)
        print("\nAttention weights (softmax):", attn_weights_qt)
