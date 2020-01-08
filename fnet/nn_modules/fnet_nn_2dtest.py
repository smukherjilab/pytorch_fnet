import fnet.nn_modules.fnet_nn_2d_params
import pdb

class Net(fnet.nn_modules.fnet_nn_2d_params.Net):
    def __init__(self):
        super().__init__(in_channels=1, out_channels=1, mult_chan=32)
