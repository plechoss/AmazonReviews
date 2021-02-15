import torch as t
import torch.nn as nn
import torch.nn.functional as F

class WordNet(nn.Module):
    '''Basic neural network structure with 2 fully connected layers. '''
    def __init__(self, params):
        super(WordNet, self).__init__()

        self.dropout = nn.Dropout(p=params['dropout_rate'])

        self.fc1 = nn.Linear(params['size'], params['n'])
        self.fc2 = nn.Linear(params['n'], 1)

    def forward(self, x):
        x = F.relu(self.fc1(self.dropout(x)))
        x = t.sigmoid(self.fc2(self.dropout(x)))

        return x
