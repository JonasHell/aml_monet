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
        def subnet_fc():
            return lambda in_channels, out_channels: nn.Sequential(
                nn.Linear(in_channels, out_channels)
            )

        input_node = Ff.InputNode(3, 5, 5)

        split = Ff.Node(input_node, Fm.Split, {})
        flatten = Ff.Node(split.out1, Fm.Flatten, {})

        flatten_fc = Ff.Node(split, Fm.Flatten, {})
        block_fc = Ff.Node(flatten_fc, Fm.GLOWCouplingBlock, {'subnet_constructor': subnet_fc()})
        
        perm = Ff.Node(block_fc, Fm.PermuteRandom, {})
        concat = Ff.Node([flatten.out0, perm.out0], Fm.Concat, {})


        output_node = Ff.OutputNode(concat)
        
        return GraphINN([input_node, split, flatten, flatten_fc, block_fc,
                            perm, concat, output_node])

net = Network()
print(net)
torch.save(net.state_dict(), 'state_dict.pth')

net2 = Network()
print(net2)
state_dict = torch.load('state_dict.pth')
net2.load_state_dict(state_dict)
