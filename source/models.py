# %%
import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm

# %%

class CondNet(nn.Module):
  #TODO
  pass

class MonetCINN_256_blocks10_nosplit(nn.Module):
    def __init__(self, learning_rate):
        super().__init__()

        self.cinn = self.create_cinn()

    def create_cinn(self):
    
        def subnet_conv(hidden_channels_1, hidden_channels_2, kernel_size):
            padding = kernel_size // 2
            return lambda in_channels, out_channels: nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels_1, kernel_size, padding=padding),
                nn.ReLU(),
                nn.Conv2d(hidden_channels_1, hidden_channels_2, kernel_size, padding=padding),
                nn.ReLU(),
                nn.BatchNorm2d(hidden_channels_2),
                nn.Conv2d(hidden_channels_2, out_channels, kernel_size, padding=padding)
            )

        def subnet_fc(hidden_channels_1, hidden_channels_2):
            return lambda in_channels, out_channels: nn.Sequential(
                nn.Linear(in_channels, hidden_channels_1),
                nn.ReLU(),
                nn.Linear(hidden_channels_1, hidden_channels_2),
                nn.ReLU(),
                nn.Linear(hidden_channels_2, out_channels)
            )

        nodes = [Ff.InputNode(3, 256, 256)]
        conditions = [] #TODO

        # add one block (3 x 256 x 256)
        subnet = subnet_conv(32, 3)
        nodes.append(Ff.node(
            nodes[-1],
            Fm.GLOWCouplingBlock,
            {'subnet_constructor': subnet, 'clamp': 2.0},
            conditions=conditions[0] #TODO
        ))

        # downsample 3 x 256 x 256 --> 12 x 128 x 128

        # am ende fc? oder muss conv dann iwie so passen dass am ende was sinvolles rauskommt
        # muss latent space gleiche dim wie original space haben
        # welche dim müssen conditions haben
        # wieviel channel in subnet?