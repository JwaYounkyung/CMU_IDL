import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Network(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout, bidirectional, num_classes):

        super(Network, self).__init__()

        # 수정
        # Adding some sort of embedding layer or feature extractor might help performance.
        # self.embedding = nn.Embedding(input_size, embedding_size)

        # TODO 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)
        self.classification = nn.Sequential(
            nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes) 
        )
        self.logSoftmax = nn.LogSoftmax(dim=1) #TODO
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lx):
        #TODO
        packed = pack_padded_sequence(x, lx, batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        x, outputs_length = pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        assert torch.equal(lx, outputs_length)
        # 수정
        # x = self.dropout(x) lstm에 이미 dropout이 있어서 여기도 해야하나?
        x = self.classification(x)
        x = self.logSoftmax(x)

        return x, outputs_length

