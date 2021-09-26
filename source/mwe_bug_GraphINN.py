# %%
from FrEIA.framework.graph_inn import GraphINN
import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.cinn = self.create_cinn()

    def create_cinn(self):   
        def subnet():
            return lambda in_channels, out_channels: nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
            )

        input_node = Ff.InputNode(16, 5, 5)

        block1 = Ff.Node(input_node, Fm.GLOWCouplingBlock, {'subnet_constructor':subnet()})
        split1 = Ff.Node(block1, Fm.Split, {})
        flatten1 = Ff.Node(split1.out1, Fm.Flatten, {})

        block2 = Ff.Node(split1, Fm.GLOWCouplingBlock, {'subnet_constructor':subnet()})
        split2 = Ff.Node(block2, Fm.Split, {})
        flatten2 = Ff.Node(split2.out1, Fm.Flatten, {})

        block3 = Ff.Node(split2, Fm.GLOWCouplingBlock, {'subnet_constructor':subnet()})
        split3 = Ff.Node(block3, Fm.Split, {})
        flatten3 = Ff.Node(split3.out1, Fm.Flatten, {})

        blocke = Ff.Node(split3, Fm.GLOWCouplingBlock, {'subnet_constructor':subnet()})
        perme = Ff.Node(blocke, Fm.PermuteRandom, {})
        flattene = Ff.Node(perme, Fm.Flatten, {})

        concat = Ff.Node([flatten1.out0, flatten2.out0, flatten3.out0, flattene.out0], Fm.Concat, {})

        output_node = Ff.OutputNode(concat)
        
        return GraphINN([input_node,
                        block1, split1, flatten1,
                        block2, split2, flatten2,
                        block3, split3, flatten3,
                        blocke, perme, flattene,
                        concat,
                        output_node])

    def forward(self, x):
        return self.cinn(x, jac=True)

    def reverse_sample(self, z):
        return self.cinn(z, rev=True)

# %%
net = Network()
x = torch.rand((1, 16, 5, 5))
z, _ = net.forward(x)
print(z.shape)
print(z)

# %%
for i in range(1000):
    print(i)
    net = Network()
    torch.save(net.state_dict(), 'state_dict.pth')

    net2 = Network()
    state_dict = torch.load('state_dict.pth')
    net2.load_state_dict(state_dict)

