# %%
import os
from FrEIA.modules.graph_topology import Split

import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
import torch.optim
import torchvision

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import config as c


"""
Final version of condition network using squeeze net
Expects 3x226x226 pixel input vectors
"""
class ConditionNet_squeeze(nn.Module):
    def __init__(self):
        super().__init__()

        class Fire(nn.Module):
            def __init__(
                self,
                inplanes: int,
                squeeze_planes: int,
                expand1x1_planes: int,
                expand3x3_planes: int
            ) -> None:
                super(Fire, self).__init__()
                self.inplanes = inplanes
                self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
                self.squeeze_activation = nn.ReLU(inplace=True)
                self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                        kernel_size=1)
                self.expand1x1_activation = nn.ReLU(inplace=True)
                self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                        kernel_size=3, padding=1)
                self.expand3x3_activation = nn.ReLU(inplace=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.squeeze_activation(self.squeeze(x))
                return torch.cat([
                    self.expand1x1_activation(self.expand1x1(x)),
                    self.expand3x3_activation(self.expand3x3(x))
                ], 1)        
        
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                # 3 x 226 x 226
                nn.Conv2d(3, 64, kernel_size=3, stride=2)
                # 64 x 112 x 112
            ),
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64)
                # 128 x 56 x 56
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128)
                # 256 x 28 x 28
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256)
                # 512 x 14 x 14
            )
        ])
        
    def forward(self, photo):
        outputs = [photo]
        for i, module in enumerate(self.res_blocks):
            #print(f"We are in layer {i}")
            outputs.append(module(outputs[-1]))
            #print(f"Output size: {outputs[-1].size()}")
        return outputs[1:]

    def initialize_pretrained(self):
        # set where the downloaded model should be saved
        torch.hub.set_dir(c.squeeze_path)
        #torch.hub.set_dir(c.vgg11_path)

        # download model if not already done
        torchvision.models.squeezenet1_1(pretrained=True)

        # load pretrained weights and biases
        pretrained_dict = torch.load(c.squeeze_path+'/checkpoints/squeezenet1_1-b8a52dc0.pth')
        pretrained_keys = list(pretrained_dict.keys())

        # initialize with pretrained weights and biases
        for i, (_, param) in enumerate(self.named_parameters()):
            param.data = pretrained_dict[pretrained_keys[i]]

"""
Final version of INN, uses squeeze net as conditioning network
Expects 3x224x224 pixel input vectors as images 3x226x226 pixels as condition
"""
class MonetCINN_squeeze(nn.Module):
    def __init__(self, learning_rate):
        super().__init__()

        self.cinn = self.create_cinn()
        self.initialize_weights()

        self.cond_net = ConditionNet_squeeze()
        self.cond_net.initialize_pretrained()

        self.trainable_parameters = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=learning_rate, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

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

        def add_stage(nodes, block_num, subnet_func, condition=None, split_nodes=None, split_sizes=None, downsample=True, prefix=''):
            """
            Convenience function that adds an entire stage to nodes.
            """
            # add specified number of blocks
            for k in range(block_num):
                subnet = subnet_func(k)
                
                # add current block
                nodes.append(Ff.Node(
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
                    Fm.Split,
                    {'section_sizes': split_sizes, 'dim': 0},
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
                    Fm.IRevNetDownsampling,
                    {},
                    name=prefix+'-down'
                ))
            
        # create nodes with input node
        nodes = [Ff.InputNode(3, 224, 224)]

        # create conditions
        condition_nodes = [ Ff.ConditionNode(64, 112, 112),
                            Ff.ConditionNode(128, 56, 56),
                            Ff.ConditionNode(256, 28, 28),
                            Ff.ConditionNode(512, 14, 14)]

        # create split_nodes
        split_nodes = []
        
        # stage 1
        # one block (3 x 224 x 224)
        # with conv3 subnet
        subnet_func = lambda _: subnet_conv(32, 64, 3)
        add_stage(nodes, 1, subnet_func,
            prefix='stage1'
        )

        # stage 2
        # one block (12 x 112 x 112)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(64, 128, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func,
            condition=condition_nodes[0],
            split_nodes=split_nodes,
            prefix='stage2'
        )

        # stage 3
        # two blocks (24 x 56 x 56)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func,
            condition=condition_nodes[1],
            split_nodes=split_nodes,
            prefix='stage3'
        )

        # stage 4
        # two blocks (48 x 28 x 28)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func,
            condition=condition_nodes[2],
            split_nodes=split_nodes,
            prefix='stage4'
        )

        # stage 5
        # two blocks (96 x 14 x 14)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func,
            condition=condition_nodes[3],
            split_nodes=split_nodes,
            split_sizes=[12, 84],
            downsample=False,
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
        # one block (flat 2352)
        # with fc subnetwork
        subnet_func = lambda _: subnet_fc(1024, 1024)
        add_stage(nodes, 1, subnet_func,
            downsample=False,
            prefix='stage6'
        )

        # concat all the splits and the output of fc part
        nodes.append(Ff.Node(
            [sn.out0 for sn in split_nodes] + [nodes[-1].out0],
            Fm.Concat,
            {'dim':0},
            name='concat'
        ))

        # add output node
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))

        return Ff.GraphINN(nodes + split_nodes + condition_nodes)

        # problem f??r bericht: beim erstellen der graphen gibt es eine randomness, beim erstellen des graphen,
        # schwierig beim speichern und laden mit state_dict(), da die namen der parameter vond er reihenfogle bah??ngen

    def forward(self, monet, photo):
        return self.cinn(monet, c=self.cond_net(photo), jac=True)

    def reverse_sample(self, z, photo):
        return self.cinn(z, c=self.cond_net(photo), rev=True)

    def forward_c_given(self, monet, c):
        return self.cinn(monet, c=c, jac=True)

    def reverse_sample_c_given(self, z, c):
        return self.cinn(z, c=c, rev=True)

    def initialize_weights(self):
        def initialize_weights_(m):
            # Conv2d layers
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                    # xavier not possible for bias

        # Xavier initialization
        self.cinn.apply(initialize_weights_)

        # initialize last conv layer of subnet with 0
        for key, param in self.cinn.named_parameters():
            split = key.split('.')

            #DEBUG
            #if param.requires_grad:   
            if len(split) > 3 and split[3][-1] == '5': # last convolution in the coeff func
                print(key)
                param.data.fill_(0.)
            
            #TODO
            #DEBUG
            # fill last fc layer with 0 manually
            if key == 'module_list.27.subnet1.4.weight' or key == 'module_list.27.subnet2.4.weight':
                print('NIIIIICEEEEE!!!!!!')
                param.data.fill_(0.)

"""
Larger version of final INN, uses squeeze net as conditioning network
Uses twice as many coupling layers
Did not turn out to perform better than much smaller INN
Expects 3x224x224 pixel input vectors as images 3x226x226 pixels as condition
"""
class MonetCINN_squeeze_large(nn.Module):
    def __init__(self, learning_rate):
        super().__init__()

        self.cinn = self.create_cinn()
        self.initialize_weights()

        self.cond_net = ConditionNet_squeeze()
        self.cond_net.initialize_pretrained()

        self.trainable_parameters = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=learning_rate, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

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

        def add_stage(nodes, block_num, subnet_func, condition=None, split_nodes=None, split_sizes=None, downsample=True, prefix=''):
            """
            Convenience function that adds an entire stage to nodes.
            """
            # add specified number of blocks
            for k in range(block_num):
                subnet = subnet_func(k)
                
                # add current block
                nodes.append(Ff.Node(
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
                    Fm.Split,
                    {'section_sizes': split_sizes, 'dim': 0},
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
                    Fm.IRevNetDownsampling,
                    {},
                    name=prefix+'-down'
                ))
            
        # create nodes with input node
        nodes = [Ff.InputNode(3, 224, 224)]

        # create conditions
        condition_nodes = [ Ff.ConditionNode(64, 112, 112),
                            Ff.ConditionNode(128, 56, 56),
                            Ff.ConditionNode(256, 28, 28),
                            Ff.ConditionNode(512, 14, 14)]

        # create split_nodes
        split_nodes = []
        
        # stage 1
        # one block (3 x 224 x 224)
        # with conv3 subnet
        subnet_func = lambda _: subnet_conv(32, 64, 3)
        add_stage(nodes, 2, subnet_func,
            prefix='stage1'
        )

        # stage 2
        # one block (12 x 112 x 112)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(64, 128, 3 if block_num%2 else 1)
        add_stage(nodes, 4, subnet_func,
            condition=condition_nodes[0],
            split_nodes=split_nodes,
            prefix='stage2'
        )

        # stage 3
        # two blocks (24 x 56 x 56)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 4, subnet_func,
            condition=condition_nodes[1],
            split_nodes=split_nodes,
            prefix='stage3'
        )

        # stage 4
        # two blocks (48 x 28 x 28)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 4, subnet_func,
            condition=condition_nodes[2],
            split_nodes=split_nodes,
            prefix='stage4'
        )

        # stage 5
        # two blocks (96 x 14 x 14)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 4, subnet_func,
            condition=condition_nodes[3],
            split_nodes=split_nodes,
            split_sizes=[12, 84],
            downsample=False,
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
        # one block (flat 2352)
        # with fc subnetwork
        subnet_func = lambda _: subnet_fc(1024, 1024)
        add_stage(nodes, 4, subnet_func,
            downsample=False,
            prefix='stage6'
        )

        # concat all the splits and the output of fc part
        nodes.append(Ff.Node(
            [sn.out0 for sn in split_nodes] + [nodes[-1].out0],
            Fm.Concat,
            {'dim':0},
            name='concat'
        ))

        # add output node
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))

        return Ff.GraphINN(nodes + split_nodes + condition_nodes)

        # problem f??r bericht: beim erstellen der graphen gibt es eine randomness, beim erstellen des graphen,
        # schwierig beim speichern und laden mit state_dict(), da die namen der parameter vond er reihenfogle bah??ngen

    def forward(self, monet, photo):
        return self.cinn(monet, c=self.cond_net(photo), jac=True)

    def reverse_sample(self, z, photo):
        return self.cinn(z, c=self.cond_net(photo), rev=True)

    def forward_c_given(self, monet, c):
        return self.cinn(monet, c=c, jac=True)

    def reverse_sample_c_given(self, z, c):
        return self.cinn(z, c=c, rev=True)

    def initialize_weights(self):
        def initialize_weights_(m):
            # Conv2d layers
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                    # xavier not possible for bias

        # Xavier initialization
        self.cinn.apply(initialize_weights_)

        # initialize last conv layer of subnet with 0
        for key, param in self.cinn.named_parameters():
            split = key.split('.')

            #DEBUG
            #if param.requires_grad:   
            if len(split) > 3 and split[3][-1] == '5': # last convolution in the coeff func
                print(key)
                param.data.fill_(0.)
            
            #TODO
            #DEBUG
            # fill last fc layer with 0 manually
            fc_blocks = ['45', '47', '49', '51']
            last_layers = ['.subnet1.4.weight', '.subnet2.4.weight']
            fc_keys = ['module_list.' + i + j for i in fc_blocks for j in last_layers]
            #if key == 'module_list.27.subnet1.4.weight' or key == 'module_list.27.subnet2.4.weight':
            if key in fc_keys:
                print('NIIIIICEEEEE!!!!!!')
                param.data.fill_(0.)




"""
First version of condition network using pretrained VGG11 from pytorch library
Expects 3x112x112 pixel input images
"""

class ConditionNet_VGG(nn.Module):
    def __init__(self):
        super().__init__()

        class Flatten(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                # 3 x 224 x 224
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 64 x 224 x 224
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                # 64 x 112 x 112
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                # 128 x 112 x 112, correct :)
            ),
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                # 128 x 56 x 56
                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 256 x 56 x 56
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                # 256 x 56 x 56, correct :)
            ),
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                # 256 x 28 x 28
                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 512 x 28 x 28
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                # 512 x 28 x 28, correct :)
            ),
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                # 512 x 14 x 14
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                # 512 x 14 x 14
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                # 512 x 14 x 14, correct :)
            ),
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                # 512 x 7 x 7, correct :)
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(7, 7)),
                
                Flatten(),
                # 512 x 7 x 7
                nn.Linear(in_features=25088, out_features=4096, bias=True),
                # 4096
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096, bias=True)
                # 4096
                #nn.ReLU(inplace=True),
                #nn.Dropout(p=0.5, inplace=False), # TODO: with dropout?
                #nn.Linear(in_features=4096, out_features=1000, bias=True)
                # 1000
            )
        ])
        
    def forward(self, photo):
        outputs = [photo]
        for i, module in enumerate(self.res_blocks):
            #print(f"We are in layer {i}")
            outputs.append(module(outputs[-1]))
            #print(f"Output size: {outputs[-1].size()}")
        return outputs[1:]

    def initialize_pretrained(self, pretrained_path):
        # set where the downloaded model should be saved
        torch.hub.set_dir(pretrained_path)
        #torch.hub.set_dir(c.vgg11_path)

        # download model if not already done
        torchvision.models.vgg11(pretrained=True)

        # load pretrained weights and biases
        pretrained_dict = torch.load(pretrained_path+'/checkpoints/vgg11-8a719046.pth')
        #pretrained_dict = torch.load(c.vgg11_path+'/checkpoints/vgg11-8a719046.pth')
        pretrained_keys = list(pretrained_dict.keys())

        # initialize with pretrained weights and biases
        for i, (key, param) in enumerate(self.named_parameters()):
            param.data = pretrained_dict[pretrained_keys[i]]

    # evtl auch m??glich:
    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # dann einfach in forward die entsprechenden unter module aufrufen, vermutlich baer ??hnlich gro??er aufwand
        
class MonetCINN_VGG(nn.Module):
    def __init__(self, learning_rate, pretrained_path=os.getcwd()):
        super().__init__()

        self.cinn = self.create_cinn()
        self.initialize_weights()

        self.cond_net = ConditionNet_VGG()
        self.cond_net.initialize_pretrained(pretrained_path)

        self.trainable_parameters = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=learning_rate, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

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

        def add_stage(nodes, block_num, subnet_func, condition=None, split_nodes=None, split_sizes=None, downsample=True, prefix=''):
            """
            Convenience function that adds an entire stage to nodes.
            """
            #TODO: does appending work correctly?

            # add specified number of blocks
            for k in range(block_num):
                subnet = subnet_func(k)
                
                # add current block
                nodes.append(Ff.Node(
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
            #print(nodes[-1])
            # split channels off
            if split_nodes is not None:
                nodes.append(Ff.Node(
                    nodes[-1],
                    Fm.Split,
                    {'section_sizes': split_sizes, 'dim': 0},
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
                    #Fm.HaarDownsampling,
                    #{'rebalance': 0.5},
                    Fm.IRevNetDownsampling,
                    {},
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
        #print(nodes[-1])
        # stage 5
        # two blocks (96 x 7 x 7)
        # one with conv1 and one with conv3 subnet
        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
        add_stage(nodes, 2, subnet_func,
            condition=condition_nodes[4],
            downsample=False,
            split_nodes=split_nodes,
            split_sizes=[24, 72],
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
        #print(nodes[-1])
        # concat all the splits and the output of fc part
        nodes.append(Ff.Node(
            [sn.out0 for sn in split_nodes] + [nodes[-1].out0],
            Fm.Concat,
            {'dim':0},
            name='concat'
        ))
        #print(nodes[-1])
        # add output node
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        #print(nodes[-1])
        #TODO: use GraphINN or ReversibleGraphNet??
        #DEBUG
        #return Ff.ReversibleGraphNet(nodes + split_nodes + condition_nodes)
        return Ff.GraphINN(nodes + split_nodes + condition_nodes)

        # problem f??r bericht: beim erstellen der graphen gibt es eine randomness, beim erstellen des graphen,
        # schwierig beim speichern und laden mit state_dict(), da die namen der parameter vond er reihenfogle bah??ngen

    def forward(self, monet, photo):
        return self.cinn(monet, c=self.cond_net(photo), jac=True)

    def reverse_sample(self, z, photo):
        return self.cinn(z, c=self.cond_net(photo), rev=True)

    def forward_c_given(self, monet, c):
        return self.cinn(monet, c=c, jac=True)

    def reverse_sample_c_given(self, z, c):
        return self.cinn(z, c=c, rev=True)

    def initialize_weights(self):
        def initialize_weights_(m):
            # Conv2d layers
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): #TODO: what exactly means xavier initialization?
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
                    # xavier not possible for bias

                '''
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
                '''

        # Xavier initialization
        self.cinn.apply(initialize_weights_)

        # initialize last conv layer of subnet with 0
        for key, param in self.cinn.named_parameters():
            split = key.split('.')
            #print(key)
            #DEBUG
            #if param.requires_grad:
                
            if len(split) > 3 and split[3][-1] == '5': # last convolution in the coeff func
                print(key)
                param.data.fill_(0.)
            
            #TODO
            #DEBUG
            # fill last fc layer with 0 manually
            #if key == 'module_list.23.subnet1.4.weight' or key == 'module_list.23.subnet2.4.weight':
            if key == 'module_list.27.subnet1.4.weight' or key == 'module_list.27.subnet2.4.weight':
                print('NIIIIICEEEEE!!!!!!')
                param.data.fill_(0.)
                

"""
Other networks used for debugging.
"""
            

#class ConditionNet_debug(nn.Module):
#    def __init__(self):
#        super().__init__()
#
#        class Flatten(nn.Module):
#            def __init__(self, *args):
#                super().__init__()
#            def forward(self, x):
#                return x.view(x.shape[0], -1)
#
#        self.res_blocks = nn.ModuleList([
#            nn.Sequential(
#                # 3 x 224 x 224
#                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                # 64 x 224 x 224
#                nn.ReLU(inplace=True),
#                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#                # 64 x 112 x 112
#                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                # 128 x 112 x 112, correct :)
#                nn.ReLU(inplace=True)
#            ),
#            nn.Sequential(
#                #nn.ReLU(inplace=True),
#                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#                # 128 x 56 x 56
#                nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                # 256 x 56 x 56
#                nn.ReLU(inplace=True),
#                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                # 256 x 56 x 56, correct :)
#                nn.ReLU(inplace=True)
#            ),
#            nn.Sequential(
#                #nn.ReLU(inplace=True),
#                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#                # 256 x 28 x 28
#                nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                # 512 x 28 x 28
#                nn.ReLU(inplace=True),
#                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                # 512 x 28 x 28, correct :)
#                nn.ReLU(inplace=True)
#            ),
#            nn.Sequential(
#                #nn.ReLU(inplace=True),
#                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#                # 512 x 14 x 14
#                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                # 512 x 14 x 14
#                nn.ReLU(inplace=True),
#                nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                # 512 x 14 x 14, correct :)
#                nn.ReLU(inplace=True)
#            ),
#            nn.Sequential(
#                #nn.ReLU(inplace=True),
#                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#                # 512 x 7 x 7, correct :)
#            ),
#            nn.Sequential(
#                nn.AdaptiveAvgPool2d(output_size=(7, 7)),
#                
#                Flatten(),
#                # 512 x 7 x 7
#                nn.Linear(in_features=25088, out_features=4096, bias=True),
#                # 4096
#                nn.ReLU(inplace=True),
#                nn.Dropout(p=0.5, inplace=False),
#                nn.Linear(in_features=4096, out_features=4096, bias=True),
#                # 4096
#                #nn.ReLU(inplace=True),
#                #nn.Dropout(p=0.5, inplace=False), # TODO: with dropout?
#                #nn.Linear(in_features=4096, out_features=1000, bias=True)
#                # 1000
#            )
#        ])
#        
#    def forward(self, photo):
#        outputs = [photo]
#        for i, module in enumerate(self.res_blocks):
#            #print(f"We are in layer {i}")
#            outputs.append(module(outputs[-1]))
#            #print(f"Output size: {outputs[-1].size()}")
#        return outputs[1:]
#
#    def initialize_pretrained(self, pretrained_path):
#        # set where the downloaded model should be saved
#        torch.hub.set_dir(pretrained_path)
#        #torch.hub.set_dir(c.vgg11_path)
#
#        # download model if not already done
#        torchvision.models.vgg11(pretrained=True)
#
#        # load pretrained weights and biases
#        pretrained_dict = torch.load(pretrained_path+'/checkpoints/vgg11-8a719046.pth')
#        #pretrained_dict = torch.load(c.vgg11_path+'/checkpoints/vgg11-8a719046.pth')
#        pretrained_keys = list(pretrained_dict.keys())
#
#        # initialize with pretrained weights and biases
#        for i, (key, param) in enumerate(self.named_parameters()):
#            param.data = pretrained_dict[pretrained_keys[i]]
#    
#    def initialize_zero(self):
#        # initialize with pretrained weights and biases
#        for i, (key, param) in enumerate(self.named_parameters()):
#            param.data.fill_(0.)
#
#class MonetCINN_debug(nn.Module):
#    def __init__(self, learning_rate, pretrained_path=os.getcwd()):
#        super().__init__()
#
#        self.cinn = self.create_cinn()
#        self.initialize_weights_priv()
#
#        #self.cond_net = ConditionNet_debug()
#        #self.cond_net.initialize_pretrained(pretrained_path)
#
#        self.trainable_parameters = [p for p in self.parameters() if p.requires_grad]
#        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=learning_rate, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
#
#    def create_cinn(self):
#    
#        def subnet_conv(hidden_channels_1, hidden_channels_2, kernel_size):
#            padding = kernel_size // 2
#            return lambda in_channels, out_channels: nn.Sequential(
#                nn.Conv2d(in_channels, hidden_channels_1, kernel_size, padding=padding),
#                nn.ReLU(),
#                nn.Conv2d(hidden_channels_1, hidden_channels_2, kernel_size, padding=padding),
#                nn.ReLU(),
#                nn.BatchNorm2d(hidden_channels_2),
#                nn.Conv2d(hidden_channels_2, out_channels, kernel_size, padding=padding)
#            )
#
#        def subnet_fc(hidden_channels_1, hidden_channels_2):
#            return lambda in_channels, out_channels: nn.Sequential(
#                nn.Linear(in_channels, hidden_channels_1),
#                nn.ReLU(),
#                nn.Linear(hidden_channels_1, hidden_channels_2),
#                nn.ReLU(),
#                nn.Linear(hidden_channels_2, out_channels)
#            )
#
#        def add_stage(nodes, block_num, subnet_func, condition=None, split_nodes=None, split_sizes=None, downsample=True, prefix=''):
#            """
#            Convenience function that adds an entire stage to nodes.
#            """
#            #TODO: does appending work correctly?
#
#            # add specified number of blocks
#            for k in range(block_num):
#                subnet = subnet_func(k)
#                
#                # add current block
#                nodes.append(Ff.Node(
#                    nodes[-1],
#                    Fm.GLOWCouplingBlock,
#                    {'subnet_constructor': subnet, 'clamp': 2.0},
#                    conditions=condition, #TODO
#                    name=prefix+f'-block{k+1}'
#                ))
#                
#                # add permutation after each block
#                nodes.append(Ff.Node(
#                    nodes[-1],
#                    Fm.PermuteRandom,
#                    {},
#                    name=prefix+f'-block{k+1}-perm'
#                ))
#                
#            #print(nodes[-1])
#            # split channels off
#            if split_nodes is not None:
#                nodes.append(Ff.Node(
#                    nodes[-1],
#                    Fm.Split,
#                    {'section_sizes': split_sizes, 'dim': 0},
#                    name=prefix+'split'
#                ))
#                
#                split_nodes.append(Ff.Node(
#                    nodes[-1].out1,
#                    Fm.Flatten,
#                    {},
#                    name=prefix+'flatten'
#                ))
#                
#
#            # add downsampling at the end of stage
#            if downsample:
#                nodes.append(Ff.Node(
#                    nodes[-1],
#                    #Fm.HaarDownsampling,
#                    #{'rebalance': 0.5},
#                    Fm.IRevNetDownsampling,
#                    {},
#                    name=prefix+'-down'
#                ))
#            
#        # create nodes with input node
#        #nodes = [Ff.InputNode(3, 256, 256)]
#        nodes = [Ff.InputNode(3, 112, 112)]
#
#        '''
#        # create conditions
#        condition_nodes = [ Ff.ConditionNode(128, 112, 112),
#                            Ff.ConditionNode(256, 56, 56),
#                            Ff.ConditionNode(512, 28, 28),
#                            Ff.ConditionNode(512, 14, 14),
#                            Ff.ConditionNode(512, 7, 7),
#                            Ff.ConditionNode(4096)] #TODO: 1000 or 4096?
#        
#        '''
#        # create split_nodes
#        split_nodes = []
#        
#
#        # stage 1
#        # one block (3 x 112 x 112)
#        # with conv3 subnet
#        subnet_func = lambda _: subnet_conv(32, 64, 3)
#        add_stage(nodes, 1, subnet_func,
#            #condition=condition_nodes[0],
#            prefix='stage1'
#        )
#
#        # stage 2
#        # two blocks (12 x 56 x 56)
#        # one with conv1 and one with conv3 subnet
#        subnet_func = lambda block_num: subnet_conv(64, 128, 3 if block_num%2 else 1)
#        add_stage(nodes, 2, subnet_func,
#            #condition=condition_nodes[1],
#            split_nodes=split_nodes,
#            prefix='stage2'
#        )
#
#        # stage 3
#        # two blocks (24 x 28 x 28)
#        # one with conv1 and one with conv3 subnet
#        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
#        add_stage(nodes, 2, subnet_func,
#            #condition=condition_nodes[2],
#            split_nodes=split_nodes,
#            prefix='stage3'
#        )
#
#        # stage 4
#        # two blocks (48 x 14 x 14)
#        # one with conv1 and one with conv3 subnet
#        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
#        add_stage(nodes, 2, subnet_func,
#            #condition=condition_nodes[3],
#            split_nodes=split_nodes,
#            prefix='stage4'
#        )
#        #TODO: does it make sense to increase num of channels in subnets?
#        #TODO: should they be larger in the beginning due to condition
#        #print(nodes[-1])
#        # stage 5
#        # two blocks (96 x 7 x 7)
#        # one with conv1 and one with conv3 subnet
#        subnet_func = lambda block_num: subnet_conv(128, 256, 3 if block_num%2 else 1)
#        add_stage(nodes, 2, subnet_func,
#            #condition=condition_nodes[4],
#            downsample=False,
#            split_nodes=split_nodes,
#            split_sizes=[24, 72],
#            prefix='stage5'
#        )
#
#        # flatten for fc part
#        nodes.append(Ff.Node(
#            nodes[-1],
#            Fm.Flatten,
#            {},
#            name='flatten'
#        ))
#
#        # stage 6
#        # one block (flat 1176)
#        # with fc subnetwork
#        subnet_func = lambda _: subnet_fc(1024, 1024)
#        add_stage(nodes, 1, subnet_func,
#            #condition=condition_nodes[5],
#            downsample=False,
#            prefix='stage6'
#        )
#        
#        #print(nodes[-1])
#        # concat all the splits and the output of fc part
#        nodes.append(Ff.Node(
#            [sn.out0 for sn in split_nodes] + [nodes[-1].out0],
#            Fm.Concat,
#            {'dim':0},
#            name='concat'
#        ))
#        
#        #print(nodes[-1])
#        # add output node
#        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
#        #print(nodes[-1])
#        #TODO: use GraphINN or ReversibleGraphNet??
#        return Ff.GraphINN(nodes + split_nodes) # + condition_nodes)
#        #return Ff.GraphINN(nodes + split_nodes + condition_nodes)
#
#        # problem f??r bericht: beim erstellen der graphen gibt es eine randomness, beim erstellen des graphen,
#        # schwierig beim speichern und laden mit state_dict(), da die namen der parameter vond er reihenfogle bah??ngen
#
#    def forward(self, monet):
#        #return self.cinn(monet, c=self.cond_net(photo), jac=True)
#        return self.cinn(monet)
#
#    def reverse_sample(self, z):
#        #return self.cinn(z, c=self.cond_net(photo), rev=True)
#        return self.cinn(z, rev=True)
#
#    '''
#    def forward_c_given(self, monet, c):
#        return self.cinn(monet, c=c, jac=True)
#
#    def reverse_sample_c_given(self, z, c):
#        return self.cinn(z, c=c, rev=True)
#    '''
#
#    def initialize_weights_priv(self):
#        def initialize_weights_priv_(m):
#            # Conv2d layers
#            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): #TODO: what exactly means xavier initialization?
#                nn.init.xavier_normal_(m.weight.data)
#                if m.bias is not None:
#                    nn.init.constant_(m.bias.data, 0)
#                    # xavier not possible for bias
#
#                '''
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.constant_(m.weight.data, 1)
#                nn.init.constant_(m.bias.data, 0)
#                '''
#
#        # Xavier initialization
#        self.cinn.apply(initialize_weights_priv_)
#
#        # initialize last conv layer of subnet with 0
#        for key, param in self.cinn.named_parameters():
#            split = key.split('.')
#            #print(key)
#            #DEBUG
#            #if param.requires_grad:
#                
#            if len(split) > 3 and split[3][-1] == '5': # last convolution in the coeff func
#                print(key)
#                param.data.fill_(0.)
#            
#            #TODO
#            #DEBUG
#            # fill last fc layer with 0 manually
#            #if key == 'module_list.23.subnet1.4.weight' or key == 'module_list.23.subnet2.4.weight':
#            if key == 'module_list.27.subnet1.4.weight' or key == 'module_list.27.subnet2.4.weight':
#                print('NIIIIICEEEEE!!!!!!')
#                param.data.fill_(0.)
#
#class MonetCINN_simple(nn.Module):
#    def __init__(self, learning_rate, pretrained_path=os.getcwd()):
#        super().__init__()
#
#        self.cinn = self.create_cinn()
#        self.initialize_weights_priv()
#
#    def create_cinn(self):
#    
#        def subnet_conv(hidden_channels_1, hidden_channels_2, kernel_size):
#            padding = kernel_size // 2
#            return lambda in_channels, out_channels: nn.Sequential(
#                nn.Conv2d(in_channels, hidden_channels_1, kernel_size, padding=padding),
#                nn.ReLU(),
#                nn.Conv2d(hidden_channels_1, hidden_channels_2, kernel_size, padding=padding),
#                nn.ReLU(),
#                nn.BatchNorm2d(hidden_channels_2),
#                nn.Conv2d(hidden_channels_2, out_channels, kernel_size, padding=padding)
#            )
#
#        def subnet_fc(hidden_channels_1, hidden_channels_2):
#            return lambda in_channels, out_channels: nn.Sequential(
#                nn.Linear(in_channels, hidden_channels_1),
#                nn.ReLU(),
#                nn.Linear(hidden_channels_1, hidden_channels_2),
#                nn.ReLU(),
#                nn.Linear(hidden_channels_2, out_channels)
#            )
#   
#        # create nodes with input node
#        nodes = [Ff.InputNode(3, 112, 112)]
#
#        subnet = subnet_conv(32, 64, 3)
#        nodes.append(Ff.Node(
#            nodes[-1],
#            Fm.GLOWCouplingBlock,
#            {'subnet_constructor': subnet, 'clamp': 2.0}
#        ))
#
#        
#        nodes.append(Ff.Node(
#            nodes[-1],
#            Fm.HaarDownsampling,
#            {}
#        ))
#        '''
#        nodes.append(Ff.Node(
#            nodes[-1],
#            Fm.IRevNetDownsampling,
#            {}
#        ))
#        '''
#        subnet = subnet_conv(32, 64, 3)
#        nodes.append(Ff.Node(
#            nodes[-1],
#            Fm.GLOWCouplingBlock,
#            {'subnet_constructor': subnet, 'clamp': 2.0}
#        ))
#
#        nodes.append(Ff.Node(
#            nodes[-1],
#            Fm.PermuteRandom,
#            {}
#        ))
#        
#        # flatten for fc part
#        nodes.append(Ff.Node(
#            nodes[-1],
#            Fm.Flatten,
#            {},
#            name='flatten'
#        ))
#
#        split_node = Ff.Node(
#            nodes[-1],
#            Fm.Split,
#            {}
#        )
#
#        nodes.append(split_node)
#
#        subnet = subnet_fc(1024, 1024)
#        nodes.append(Ff.Node(
#            nodes[-1],
#            Fm.GLOWCouplingBlock,
#            {'subnet_constructor': subnet, 'clamp': 2.0}
#        ))
#        
#        nodes.append(Ff.Node(
#            [nodes[-1].out0, split_node.out1],
#            Fm.Concat,
#            {}
#        ))
#
#        # add output node
#        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
#
#        return Ff.GraphINN(nodes)
#
#    def forward(self, monet):
#        #return self.cinn(monet, c=self.cond_net(photo), jac=True)
#        return self.cinn(monet)
#
#    def reverse_sample(self, z):
#        #return self.cinn(z, c=self.cond_net(photo), rev=True)
#        return self.cinn(z, rev=True)
#
#    def initialize_weights_priv(self):
#        def initialize_weights_priv_(m):
#            # Conv2d layers
#            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): #TODO: what exactly means xavier initialization?
#                nn.init.xavier_normal_(m.weight.data)
#                if m.bias is not None:
#                    nn.init.constant_(m.bias.data, 0)
#                    # xavier not possible for bias
#
#                '''
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.constant_(m.weight.data, 1)
#                nn.init.constant_(m.bias.data, 0)
#                '''
#
#        # Xavier initialization
#        self.cinn.apply(initialize_weights_priv_)
#
#        # initialize last conv layer of subnet with 0
#        for key, param in self.cinn.named_parameters():
#            split = key.split('.')
#            #print(key)
#            #DEBUG
#            #if param.requires_grad:
#                
#            if len(split) > 3 and split[3][-1] == '5': # last convolution in the coeff func
#                print(key)
#                param.data.fill_(0.)
#            
#            #TODO
#            #DEBUG
#            # fill last fc layer with 0 manually
#            #if key == 'module_list.23.subnet1.4.weight' or key == 'module_list.23.subnet2.4.weight':
#            if key == 'module_list.6.subnet1.4.weight' or key == 'module_list.6.subnet2.4.weight':
#                print('NIIIIICEEEEE!!!!!!')
#                param.data.fill_(0.)