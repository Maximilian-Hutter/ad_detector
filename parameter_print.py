from basic_models import *
import numpy as np
import torch
import torch.nn as nn
import socket
import torch.backends.cudnn as cudnn
from torchsummary import summary
import time

if __name__ == "__main__":
    hparams = {
        "seed": 123,
        "gpus": 1,
        "height":None,
        "width":None,
    }

    np.random.seed(hparams["seed"])    # set seed to default 123 or opt
    torch.manual_seed(hparams["seed"])
    torch.cuda.manual_seed(hparams["seed"])
    gpus_list = range(hparams["gpus"])
    hostname = str(socket.gethostname)
    cudnn.benchmark = True

    # defining shapes


    Net = None


    start = time.time()
    summary(Net, (3, 640, 360))
    end = time.time()

    proctime = end-start
    print(proctime)

    # pytorch_params = sum(p.numel() for p in Net.parameters())
    # print("Network parameters: {}".format(pytorch_params))

    # def print_network(net):
    #     num_params = 0
    #     for param in net.parameters():
    #         num_params += param.numel()
    #     print(net)
    #     print('Total number of parameters: %d' % num_params)

    # print('===> Building Model ')
    # Net = Net


    # print('----------------Network architecture----------------')
    # print_network(Net)
    # print('----------------------------------------------------')