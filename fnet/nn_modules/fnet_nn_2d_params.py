import torch
import pdb

class Net(torch.nn.Module):
    def __init__(self,
                 depth=4,
                 mult_chan=32,
                 in_channels=1,
                 out_channels=1,
    ):
        super().__init__()
        self.depth = depth
        # print(f"fnet/nn_module/fnet_nn_2d5i6o: depth={self.depth}")
        self.mult_chan = mult_chan
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.net_recurse = _Net_recurse(n_in_channels=self.in_channels, mult_chan=self.mult_chan, depth=self.depth)
        self.conv_out = torch.nn.Conv2d(self.mult_chan*self.in_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # print(f"FLAG 1: n_in_channels={self.in_channels}, mult_chan={self.mult_chan}, depth={self.depth}")
        x_rec = self.net_recurse(x)
        return self.conv_out(x_rec)


class _Net_recurse(torch.nn.Module):
    def __init__(self, n_in_channels, mult_chan=2, depth=0):
        """Class for recursive definition of U-network.p

        Parameters:
        in_channels - (int) number of channels for input.
        mult_chan - (int) factor to determine number of output channels
        depth - (int) if 0, this subnet will only be convolutions that double the channel count.
        """
        super().__init__()
        self.depth = depth
        n_out_channels = n_in_channels*mult_chan
        self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels)
        
        if depth > 0:
            self.sub_2conv_less = SubNet2Conv(2*n_out_channels, n_out_channels)
            self.conv_down = torch.nn.Conv2d(n_out_channels, n_out_channels, 2, stride=2)
            self.bn0 = torch.nn.BatchNorm2d(n_out_channels)
            self.relu0 = torch.nn.ReLU()
            
            self.convt = torch.nn.ConvTranspose2d(2*n_out_channels, n_out_channels, kernel_size=2, stride=2)
            self.bn1 = torch.nn.BatchNorm2d(n_out_channels)
            self.relu1 = torch.nn.ReLU()
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth=(depth - 1))
            
    def forward(self, x):
        if self.depth == 0:
            # print(f"FLAG 2: size of x is {x.size()}")
            return self.sub_2conv_more(x)
        else:  # depth > 0
            # print(f"FLAG 3: size of x is {x.size()}")
            x_2conv_more = self.sub_2conv_more(x)
            # print(f"FLAG 4: size of x_2conv_more is {x_2conv_more.size()}")
            x_conv_down = self.conv_down(x_2conv_more)
            # print(f"FLAG 5: size of x_conv_down is {x_conv_down.size()}")
            x_bn0 = self.bn0(x_conv_down)
            # print(f"FLAG 6: size of x_bn0 is {x_bn0.size()}")
            x_relu0 = self.relu0(x_bn0)
            # print(f"FLAG 7: size of x_relu0 is {x_relu0.size()}")
            x_sub_u = self.sub_u(x_relu0)
            # print(f"FLAG 8: size of x_sub_u is {x_sub_u.size()}")
            x_convt = self.convt(x_sub_u)
            # print(f"FLAG 9: size of x_convt is {x_convt.size()}")
            x_bn1 = self.bn1(x_convt)
            # print(f"FLAG 10: size of x_bn1 is {x_bn1.size()}")
            x_relu1 = self.relu1(x_bn1)
            # print(f"FLAG 11")
            x_cat = torch.cat((x_2conv_more, x_relu1), 1)  # concatenate
            # print(f"FLAG 12: size of x_cat is {x_cat.size()}")
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less

class SubNet2Conv(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(n_in,  n_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(n_out)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(n_out)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        # print(f"FLAG 13: size of x is {x.size()}")
        x = self.conv1(x)
        # print(f"FLAG 14: size of x is {x.size()}")
        x = self.bn1(x)
        # print(f"FLAG 15: size of x is {x.size()}")
        x = self.relu1(x)
        # print(f"FLAG 16: size of x is {x.size()}")
        x = self.conv2(x)
        # print(f"FLAG 17: size of x is {x.size()}")
        x = self.bn2(x)
        # print(f"FLAG 18: size of x is {x.size()}")
        x = self.relu2(x)
        return x
