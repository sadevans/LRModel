import torch.nn as nn
import torch


class TransformerEncoder(nn.Module):
    def __init__(self, num_channels=384, num_heads=8, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(num_channels, num_heads, dropout=dropout)

        # Feedforward Network
        self.linear1 = nn.Linear(num_channels, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, num_channels)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, x_mask=None, x_key_padding_mask=None):
        
        attn_output, _ = self.self_attn(x, x, x, attn_mask=x_mask, key_padding_mask=x_key_padding_mask)
        x = x + self.dropout1(attn_output)  # Add & Norm after residual connection here
        x = self.norm1(x)

        x2 = self.linear2(self.relu(self.linear1(x)))
        #print(x2.shape, x.shape)
        x = x + self.dropout2(x2)  # Add & Norm after residual connection here
        x = self.norm2(x)

        return x