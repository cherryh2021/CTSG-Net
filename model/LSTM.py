import torch
import torch.nn as nn
import torch.nn.functional as F


class VertexLSTM(nn.Module):
    def __init__(self, feature_num, hidden_dim, sequence_out, num_layers, dropout_rate):
        super(VertexLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.sequence_out = sequence_out
        self.num_layers = num_layers

        # LSTM层，每个顶点的时间序列独立处理
        self.lstm = nn.LSTM(feature_num, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        
        # 全连接层1，从隐藏状态映射到中间特征
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

        # 全连接层2，从中间特征映射到最终的时间步长
        self.fc2 = nn.Linear(hidden_dim, feature_num * sequence_out)

    def forward(self, x):
        # x的维度是[batch_size, feature_num, sequence_in, vertex_num]
        batch_size, feature_num, sequence_in, vertex_num = x.shape

        # 重塑x以将每个顶点的时间序列作为独立的样本处理
        x = x.permute(0, 3, 2, 1)  # 重塑为[batch_size, vertex_num, sequence_in, feature_num]
        x = x.reshape(batch_size * vertex_num, sequence_in, feature_num)  # 合并批次和顶点维度

        # LSTM前向传播
        h0 = torch.zeros(self.num_layers, batch_size * vertex_num, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size * vertex_num, self.hidden_dim).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 使用最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 通过全连接层1
        #fc1_out = F.relu(self.fc1(lstm_out))

        # 通过全连接层2，映射到最终的时间步长
        fc2_out = self.fc2(lstm_out)  # [batch_size * vertex_num, feature_num * sequence_out]

        # 重塑输出以匹配期望的维度
        out = fc2_out.reshape(batch_size, vertex_num, self.sequence_out, feature_num).permute(0, 2, 3, 1)
        #print("out.shape", out.shape)
        # 输出维度为[batch_size, feature_num, sequence_out, vertex_num]
        return out
