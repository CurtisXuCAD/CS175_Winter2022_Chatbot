import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)        #first layer
        self.l2 = nn.Linear(hidden_size, hidden_size)       #second layer
        self.l3 = nn.Linear(hidden_size, output_size)       #output layer(third layer)
        self.relu = nn.ReLU()                               #activation function

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax for l3, because later applying cross entropy, it will do
        return out 