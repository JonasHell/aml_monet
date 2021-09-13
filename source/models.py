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

        def add_stage(nodes, block_num, subnet_func, condition=None, downsample=True, prefix=''):
            """
            Convenience function that adds an entire stage to nodes.
            """
            #TODO: does appending work correctly?

            # add specified number of blocks
            for k in range(block_num):
                subnet = subnet_func(block_num)
                
                # add current block
                nodes.append(Ff.node(
                    nodes[-1],
                    Fm.GLOWCouplingBlock,
                    {'subnet_constructor': subnet, 'clamp': 2.0},
                    conditions=condition, #TODO
                    name=prefix+f'-block{k+1}'
                ))

                # add permutation after each block
                nodes.append(Ff.Node(
                    nodes[-1],
                    Fm.PermuteRandom,
                    {},
                    name=prefix+f'-block{k+1}-perm'
                ))

            # add downsampling at the end of stage
            if downsample:
                nodes.append(Ff.Node(
                    nodes[-1],
                    Fm.HaarDownsampling,
                    {'rebalance':0.5},
                    name=prefix+'-down'
                ))
            
        # create nodes with input node
        nodes = [Ff.InputNode(3, 256, 256)]

        # create conditions
        condition_nodes = [] #TODO
        
        # stage 1
        # one block (3 x 256 x 256)
        # with conv3 subnet
        subnet_func = lambda _: subnet_conv(32, 64, 3)
        add_stage(nodes, 1, subnet_func, condition_nodes[0], prefix='stage1')

        # stage 2
        # two blocks (12 x 128 x 128)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(64, 128, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func, condition_nodes[1], prefix='stage2')

        # stage 3
        # two blocks (48 x 64 x 64)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func, condition_nodes[2], prefix='stage3')

        # stage 4
        # two blocks (192 x 32 x 32)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1) #TODO: does it make sense to increase num of channels in subnets?
        add_stage(nodes, 2, subnet_func, condition_nodes[3], prefix='stage4')

        # stage 5
        # two blocks (768 x 16 x 16)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func, condition_nodes[4], prefix='stage5')

        # flatten for fc part
        nodes.append(Ff.Node(
            nodes[-1],
            Fm.Flatten,
            {},
            name='flatten'
        ))

        # stage 6
        # one block (flat 196608)
        # with fc subnetwork
        subnet_func = lambda _: subnet_fc(196608, 196608) #TODO: does this make sense?
        add_stage(nodes, 1, subnet_func, condition_nodes[4], downsample=False, prefix='stage6')

        # add output node
        nodes.append(Ff.OutputNode(nodes[-1]))

        #TODO: use GraphINN or ReversibleGraphNet??
        return Ff.ReversibleGraphNet(nodes + condition_nodes)

    def forward(self, monet, photo):
        z = self.cinn(monet, c=self.cond_net(photo))
        jac = self.cinn.log_jacobian(run_forward=False)
        return z, jac

    def reverse_sample(self, z, photo):
        return self.cinn(z, c=self.cond_net(photo), rev=True)

        # am ende fc? oder muss conv dann iwie so passen dass am ende was sinvolles rauskommt
        # muss latent space gleiche dim wie original space haben
        # welche dim müssen conditions haben
        # wieviel channel in subnet?