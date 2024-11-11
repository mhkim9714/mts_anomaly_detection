import torch
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1, bidirectional=False)

    def forward(self, x):
        outputs, (hidden, cell) = self.encoder_lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        return (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1, bidirectional=False)
        self.relu = nn.ReLU()
        self.final_fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, hidden):
        output, (hidden, cell) = self.decoder_lstm(x, hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        reconstruction = self.final_fc(output)
        return reconstruction, (hidden, cell)
    

class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length, num_layers):
        super(LSTMAutoEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim        
        self.seq_len = seq_length
        self.num_layers = num_layers

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.reconstruct_decoder = Decoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, src, teacher_forcing_ratio=0):
        batch_size, sequence_length, var_length = src.size()

        encoder_hidden = self.encoder(src)
        
        inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        reconstruct_output = []
        temp_input = torch.zeros((batch_size, 1, var_length), dtype=torch.float).to(src.device)
        hidden = encoder_hidden

        for t in range(sequence_length):
            temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
            reconstruct_output.append(temp_input)

            ## Teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                temp_input = src[:,sequence_length-1-t,:].unsqueeze(1)

        reconstruct_output = torch.cat(reconstruct_output, dim=1)[:, inv_idx, :]
        return reconstruct_output
    
