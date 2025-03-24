import torch
import torch.nn as nn
import torch.nn.functional as F


class VertexLSTM(nn.Module):
    def __init__(self, feature_num, hidden_dim, sequence_out, num_layers, dropout_rate):
        super(VertexLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.sequence_out = sequence_out
        self.num_layers = num_layers

        self.lstm = nn.LSTM(feature_num, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, feature_num * sequence_out)

    def forward(self, x):
        batch_size, feature_num, sequence_in, vertex_num = x.shape  # [batch_size, feature_num, sequence_in, vertex_num]

        x = x.permute(0, 3, 2, 1)  # [batch_size, vertex_num, sequence_in, feature_num]
        x = x.reshape(batch_size * vertex_num, sequence_in, feature_num) 

        h0 = torch.zeros(self.num_layers, batch_size * vertex_num, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size * vertex_num, self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))

        lstm_out = lstm_out[:, -1, :]
        fc2_out = self.fc2(lstm_out)  # [batch_size * vertex_num, feature_num * sequence_out]
        out = fc2_out.reshape(batch_size, vertex_num, self.sequence_out, feature_num).permute(0, 2, 3, 1)
        return out  # [batch_size, feature_num, sequence_out, vertex_num]
