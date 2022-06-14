import torch.nn as nn
import torch
import torch.nn.functional as F

class Q_Network(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Q_Network, self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    def forward(self,input):
        input = torch.tensor(input).to(torch.float32)
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out =self.predict(out)

        return out