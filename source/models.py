# %%
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm

# %%

class ConditionNet(nn.Module):
    def __ini__(self):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        self.modules = nn.ModuleList([
            nn.Sequential(
                # 3 x 224 x 224
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 64 x 224 x 224
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                # 64 x 112 x 112
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 128 x 112 x 112
            ),
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                # 128 x 56 x 56
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 256 x 56 x 56
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 256 x 56 x 56
            ),
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                # 256 x 28 x 28
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 512 x 28 x 28
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 512 x 28 x 28
            ),
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                # 512 x 14 x 14
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 512 x 14 x 14
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 512 x 14 x 14
            ),
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                # 512 x 7 x 7
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(7, 7)),
                # 25088
                nn.Linear(in_features=25088, out_features=4096, bias=True),
                # 4096
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                # 4096
                #nn.ReLU(inplace=True),
                #nn.Dropout(p=0.5, inplace=False), # TODO: with dropout?
                #nn.Linear(in_features=4096, out_features=1000, bias=True)
                # 1000
            )
        ])
        
    def forward(self, photo):
        outputs = [photo]
        for module in self.modules:
            outputs.append(module(outputs[-1]))
        return outputs[1:]

    # evtl auch möglich:
    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # dann einfach in forward die entsprechenden unter module aufrufen, vermutlich baer ähnlich großer aufwand
        

class MonetCINN_112_blocks10(nn.Module):
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

        def add_stage(nodes, block_num, subnet_func, condition=None, split_nodes=None, split_sizes=[1, 1], downsample=True, prefix=''):
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

            # split channels off
            if split_nodes is not None:
                nodes.append(Ff.Node(
                    nodes[-1],
                    Fm.Split1D,
                    {'split_size_or_sections':split_sizes, 'dim':0},
                    name=prefix+'split'
                ))
                split_nodes.append(Ff.Node(
                    nodes[-1].out1,
                    Fm.Flatten,
                    {},
                    name=prefix+'flatten'
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
        #nodes = [Ff.InputNode(3, 256, 256)]
        nodes = [Ff.InputNode(3, 112, 112)]

        # create conditions
        condition_nodes = [ Ff.ConditionNode(128, 112, 112),
                            Ff.ConditionNode(256, 56, 56),
                            Ff.ConditionNode(512, 28, 28),
                            Ff.ConditionNode(512, 14, 14),
                            Ff.ConditionNode(512, 7, 7),
                            Ff.ConditionNode(4096)] #TODO: 1000 or 4096?

        # create split_nodes
        split_nodes = []
        
        # stage 1
        # one block (3 x 112 x 112)
        # with conv3 subnet
        subnet_func = lambda _: subnet_conv(32, 64, 3)
        add_stage(nodes, 1, subnet_func,
            condition=condition_nodes[0],
            prefix='stage1'
        )

        # stage 2
        # two blocks (12 x 56 x 56)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(64, 128, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func,
            condition=condition_nodes[1],
            split_nodes=split_nodes,
            prefix='stage2'
        )

        # stage 3
        # two blocks (24 x 28 x 28)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func,
            condition=condition_nodes[2],
            split_nodes=split_nodes,
            prefix='stage3'
        )

        # stage 4
        # two blocks (48 x 14 x 14)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func,
            condition=condition_nodes[3],
            split_nodes=split_nodes,
            prefix='stage4'
        )
        #TODO: does it make sense to increase num of channels in subnets?
        #TODO: should they be larger in the beginning due to condition

        # stage 5
        # two blocks (96 x 7 x 7)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func,
            condition=condition_nodes[4],
            split_nodes=split_nodes,
            split_sizes=[1, 3],
            prefix='stage5'
        )

        # flatten for fc part
        nodes.append(Ff.Node(
            nodes[-1],
            Fm.Flatten,
            {},
            name='flatten'
        ))

        # stage 6
        # one block (flat 1176)
        # with fc subnetwork
        subnet_func = lambda _: subnet_fc(1024, 1024)
        add_stage(nodes, 1, subnet_func,
            condition=condition_nodes[5],
            downsample=False,
            prefix='stage6'
        )

        # concat all the splits and the output of fc part
        nodes.append(Ff.Node(
            [sn.out0 for sn in split_nodes] + [nodes[-1].out0],
            Fm.Concat1d,
            {'dim':0},
            name='concat'
        ))

        # add output node
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))

        #TODO: use GraphINN or ReversibleGraphNet??
        return Ff.ReversibleGraphNet(nodes + split_nodes + condition_nodes)

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