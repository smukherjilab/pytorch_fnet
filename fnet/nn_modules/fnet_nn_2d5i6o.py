import fnet.nn_modules.fnet_nn_2d_params
import pdb

class Net(fnet.nn_modules.fnet_nn_2d_params.Net):
    def __init__(self):
        super().__init__(in_channels=5, out_channels=6, mult_chan=5)
