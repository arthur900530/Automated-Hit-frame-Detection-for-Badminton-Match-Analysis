import torch
import torch.nn as nn
import math

class CoordinateEmbedding(nn.Module):
    def __init__(self, in_channels: int, emb_size: int):
        super().__init__()
        half_emb = int(emb_size / 2)
        self.projection1 = nn.Linear(in_channels, half_emb)
        self.projection1_2 = nn.Linear(half_emb, half_emb)
        self.projection2 = nn.Linear(in_channels, half_emb)
        self.projection2_2 = nn.Linear(half_emb, half_emb)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        p1 = x.select(2, 0)
        p2 = x.select(2, 1)
        p1 = self.projection1(p1)
        p2 = self.projection2(p2)
        p1 = self.projection1_2(p1)
        p2 = self.projection2_2(p2)
        projected = torch.cat((p1, p2), 2)
        return projected


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5  10,1
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])



