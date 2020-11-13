import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, bidirectional=True, dropout=0):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, bidirectional=bidirectional, dropout=dropout) #输入数据2个特征维度，6个隐藏层维度，2个LSTM串联，第二个LSTM接收第一个的计算结果

        if bidirectional:
            hidden_size = hidden_size * 2
        else:
            hidden_size = hidden_size
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x1, _ = self.lstm(x)
        a, b, c = x1.shape
        out = self.out(x1.view(-1, c))
        out1 = out.view(a, b, -1)
        return out1