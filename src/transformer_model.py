import torch.nn as nn

from positional_encoder import PositionalEncoder

class TransformerModel(nn.Module):
    """
    PyTorch Lightning module for a generic Transformer model.
    """
    def __init__(self, input_dim=31, model_dim=256, num_heads=16, num_layers=2, output_dim=6, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoder(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.criterion = nn.MSELoss()
        self.decoder = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x
