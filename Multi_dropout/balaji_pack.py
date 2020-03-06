import torch
import torch.nn as nn
import torch.nn.functional as F
from activations_full import *

activ_list =[   nn.ReLU(inplace = True),
                Class_GeneralRelu(),
                nn.ELU(inplace = True),
                nn.SELU(inplace=True),
                nn.CELU(inplace=True),
                nn.LeakyReLU(0.01,inplace = True),
                nn.PReLU(),
                nn.PReLU(init=0.01),
                Class_TRelu(),
                nn.RReLU(0.1, 0.3,inplace = True)
            ]

def change_activ(model, balaji_number=0):
    for child_name, child in model.named_children():
        # balaji_number = 1
        activ = activ_list[balaji_number]
        # activ =nn.ReLU(inplace = True)
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, activ)
        else:
            change_activ(child, balaji_number)
