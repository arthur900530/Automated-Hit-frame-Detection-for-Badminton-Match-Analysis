import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from layers import PositionalEncoding, CoordinateEmbedding
import numpy as np
import math
import pickle

class OptimusPrime(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, dim_feedforward, dropout_p=0):
        super().__init__()
        # INFO
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout_p=dropout_p, max_len=600)

        self.xy_embedding = CoordinateEmbedding(in_channels=24, emb_size=dim_model)

        encoder_layers = TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout_p)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        self.decoder1 = nn.Linear(dim_model, dim_model)
        self.decoder2 = nn.Linear(dim_model, num_tokens)

    def forward(self, src, src_pad_mask=None):
        src = self.xy_embedding(src)
        src = self.positional_encoder(src) * math.sqrt(self.dim_model)
        src = src.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        output = self.transformer_encoder(src, src_key_padding_mask=src_pad_mask)
        output = F.relu(self.decoder1(output))
        output = self.decoder2(output)
        return output

    def create_src_pad_mask(self, matrix: torch.tensor, PAD_array=np.zeros((1, 2, 12, 2))) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        src_pad_mask = []
        PAD_array = torch.tensor(PAD_array).squeeze(0).to(device)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                a = matrix[i][j]
                src_pad_mask.append(torch.equal(a, PAD_array))
        src_pad_mask = torch.tensor(src_pad_mask).unsqueeze(0).reshape(matrix.shape[0], -1).to(device)
        return src_pad_mask


class OptimusPrimeContainer(object):
    """
    Optimus Prime Model Container.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.setup_model(args['opt_path'])
        self.setup_scalar(args['sc_path'])
    
    def setup_model(self, path):
        self.model = OptimusPrime(
            num_tokens=4, dim_model=512, num_heads=8, num_encoder_layers=8, dim_feedforward=1024,
            dropout_p=0
        ).to(self.device)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
    
    def setup_scalar(self, path):
        self.scaler = pickle.load(open(path, 'rb'))


    def predict(self, input_sequence):
        # Get source mask
        src_pad_mask = self.model.create_src_pad_mask(input_sequence)
        pred = self.model(input_sequence, src_pad_mask=src_pad_mask)
        pred_indices = torch.max(pred.detach(), 2).indices.squeeze(-1)

        return pred_indices