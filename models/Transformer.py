import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model) # [max_len, 1, h]
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x): # [b,t,h]
        x = x.permute(1,0,2) # [t,b,h]
        x = x + self.pe[:x.size(0)] # [t,b,h]
        x = self.dropout(x).permute(1,0,2)
        return x # [b,t,h]


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, t_past, t_future, dropout: float = 0.1):
        super().__init__()

        self.d_model = hidden_dim
        self.t_past = t_past
        self.t_future = t_future

        self.input_projection = nn.Linear(input_dim, hidden_dim) # N->h
        self.output_projection = nn.Linear(hidden_dim, input_dim) # h->N

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, src, tgt):
        src = self.pos_encoder(self.input_projection(src))
        tgt = self.pos_encoder(self.input_projection(tgt))

        # Transformer encoder
        feat = self.encoder(src)

        prediction = self.decoder(tgt, feat)
        output = self.output_projection(prediction)

        return output
    
