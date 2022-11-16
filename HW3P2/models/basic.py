import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Network(nn.Module):

    def __init__(self, input_size, embedding_size1, embedding_size2, embedding_size3, hidden_size, num_layers, dropout, bidirectional, num_classes):

        super(Network, self).__init__()

        # 수정
        self.embedding1 = nn.Sequential(
            nn.Conv1d(input_size, embedding_size1, 3, 1, padding=1), # in_channel, out_channel, kernel_size, stride
            nn.BatchNorm1d(num_features=embedding_size1),
            nn.PReLU(num_parameters=embedding_size1),
            nn.Dropout(dropout)
        )
        self.embedding2 = nn.Sequential(
            nn.Conv1d(embedding_size1, embedding_size2, 3, 1, padding=1), # in_channel, out_channel, kernel_size, stride
            nn.BatchNorm1d(num_features=embedding_size2),
            nn.PReLU(num_parameters=embedding_size2),
            nn.Dropout(dropout)
        )
        # self.embedding3 = nn.Sequential(
        #     nn.Conv1d(embedding_size2, embedding_size3, 3, 1, padding=1), # in_channel, out_channel, kernel_size, stride
        #     nn.BatchNorm1d(num_features=embedding_size3),
        #     nn.PReLU(num_parameters=embedding_size3),
        #     nn.Dropout(dropout)
        # )

        # TODO 
        self.lstm = nn.LSTM(embedding_size2, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 128),
        )
        self.fc1_norm = nn.Sequential(
            nn.BatchNorm1d(num_features=128),
            nn.PReLU(num_parameters=128),
            nn.Dropout(dropout)
        )
        self.classification = nn.Sequential(
            nn.Linear(128, num_classes) 
        )
        self.logSoftmax = nn.LogSoftmax(dim=2) #TODO
        

    def forward(self, x, lx):
        #TODO
        x = x.permute(0, 2, 1)
        x = self.embedding1(x)
        x = self.embedding2(x)
        #x = self.embedding3(x)
        x = x.permute(0, 2, 1)

        packed = pack_padded_sequence(x, lx.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        x, outputs_length = pad_packed_sequence(output, batch_first=True, total_length=x.shape[1])
        assert torch.equal(lx.cpu(), outputs_length)
        # 수정
        # x = self.dropout(x) lstm에 이미 dropout이 있어서 여기도 해야하나?
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.fc1_norm(x)
        x = x.permute(0, 2, 1)
        x = self.classification(x)
        x = self.logSoftmax(x)

        return x, outputs_length

