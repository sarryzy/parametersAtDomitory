import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.attention = nn.Softmax(dim=-1)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]

        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        Q = Q.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = self.attention(energy)

        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(N, -1, self.d_model)

        out = self.out(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, hidden_dim)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        ffn = self.ffn(x)
        x = self.norm2(ffn + x)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, num_layers):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, hidden_dim))

    def forward(self, value, key, query, mask):
        for layer in self.layers:
            query = layer(value, key, query, mask)
        return query

class VectorMappingModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, hidden_dim, num_layers):
        super(VectorMappingModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = Transformer(d_model, num_heads, hidden_dim, num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.embedding(x)  # (batch_size, d_model)
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)

        mask = None  # No mask is needed in this case
        x = self.transformer(x, x, x, mask)  # (batch_size, 1, d_model)

        x = x.squeeze(1)  # (batch_size, d_model)
        x = self.fc(x)  # (batch_size, output_dim)
        return x

input_dim = 5
output_dim = 3
d_model = 64
num_heads = 4
hidden_dim = 128
num_layers = 2

model = VectorMappingModel(input_dim, output_dim, d_model, num_heads, hidden_dim, num_layers)

# Example input vector
input_vector = torch.randn(1, input_dim)

# Get the output vector
output_vector = model(input_vector)