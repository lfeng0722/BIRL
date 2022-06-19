import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):

    def __init__(self,n_input,n_hidden,n_output):
        super(model, self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden1.weight.data.normal_(0, 0.1)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.hidden2.weight.data.normal_(0, 0.1)
        self.predict = nn.Linear(n_hidden,n_output)
        self.predict.weight.data.normal_(0, 0.1)
    def forward(self,input):
        input = torch.tensor(input).to(torch.float32)
        out = self.hidden1(input)
        out = F.elu(out)
        out = self.hidden2(out)
        out = F.elu(out)
        out =self.predict(out)
        # out = torch.abs(out)

        return out

    # def __init__(self, input_dim, output_dim, hidden_dim=128):
    #     super(model, self).__init__()
    #     self.input_dim = input_dim
    #     self.output_dim = output_dim
    #     self.hidden_dim = hidden_dim
    #
    #     self.mlp_layer = nn.Sequential(
    #         nn.Linear(self.input_dim, self.hidden_dim),
    #         nn.ReLU(),
    #         nn.Linear(self.hidden_dim, self.hidden_dim),
    #         nn.ReLU(),
    #         nn.Linear(self.hidden_dim, self.output_dim),
    #         nn.Softmax(dim=-1)
    #     )
    #
    #     self.log_probs = []
    #     self.rewards = []
    #
    # def forward(self, input):
    #     return self.mlp_layer(input)

    # def act(self, input):
    #     prob = self.forward(input)
    #     dist = torch.distributions.Categorical(prob)
    #     action = dist.sample()
    #     self.log_probs.append(dist.log_prob(action))
    #     return action.detach().item()