# %% minimal working example for HaarDownsampling bug
import torch
import numpy as np

import FrEIA.framework as Ff
import FrEIA.modules as Fm

class Net(torch.nn.Module):
    def __init__(self, down_method):
        super().__init__()

        input_node = Ff.InputNode(2, 2, 2)
        down_node = Ff.Node(input_node, down_method, {})
        output_node = Ff.OutputNode(down_node)

        self.cinn = Ff.GraphINN([input_node, down_node, output_node])

    def forward(self, x):
        return self.cinn(x)

    def reverse_sample(self, z):
        return self.cinn(z, rev=True)


net_haar = Net(Fm.HaarDownsampling)
net_irev = Net(Fm.IRevNetDownsampling)

x = torch.rand((1, 2, 2, 2))
print('input')
print('x =\n ', x.flatten(), '\n')

z_haar, _ = net_haar.forward(x)
rec_haar, _ = net_haar.reverse_sample(z_haar)
print('HaarDownsampling')
print('z =\n ', z_haar.flatten())
print('x_rec =\n ', rec_haar.flatten())
print('max_diff =\n ', np.max(rec_haar.detach().numpy().flatten()
    - x.detach().numpy().flatten()), '\n')

z_irev, _ = net_irev.forward(x)
rec_irev, _ = net_irev.reverse_sample(z_irev)
print('IRevNetDownsampling')
print('z =\n ', z_irev.flatten())
print('x_rec =\n ', rec_irev.flatten())
print('max_diff =\n ', np.max(rec_irev.detach().numpy().flatten()
    - x.detach().numpy().flatten()), '\n')
