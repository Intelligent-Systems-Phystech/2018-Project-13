import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F


CHARGES = [0., 2., 5., 8., 9., 18.]


class BaselineRegressor(nn.Module):
    def __init__(self):
        super(BaselineRegressor, self).__init__()

        self.n_atoms = 50
        self.n_types = 10

        # number of hidden units in the output regression
        self.n_hidden = 50
        self.n_hidden2 = 20

        self.W_h = nn.Linear(self.n_types, self.n_hidden)
        self.W_h2 = nn.Linear(self.n_hidden, self.n_hidden2)
        self.W_t = nn.Linear(self.n_hidden2, 1)

    def forward(self, x):
        '''
            x: [batch, n_atoms, n_types, beta, alpha]
            types: [batch, n_atoms, n_types]
        '''

        # get charge
        z = torch.autograd.Variable(
                torch.from_numpy(np.array(CHARGES))
            ).view(1, -1).float().cuda()

        # get atom frequency per molecule
        x = torch.sum(x, dim=1)
        x[:, 0] = 0

        # multiply frequency by charge
        # TODO: concatenate instead?
        z = z.expand_as(x)
        x = x * z

        # simple transform
        x = self.W_h(x)
        x = F.relu(x)
        x = self.W_h2(x)
        x = F.relu(x)
        x = self.W_t(x)

        return x
