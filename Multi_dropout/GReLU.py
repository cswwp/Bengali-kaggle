import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter

class Class_GeneralRelu(nn.Module):
	def __init__(self, leak=None, sub=None, maxv=None):
		super().__init__()
		self.leak,self.sub,self.maxv = leak,sub,maxv

	def forward(self, x):
		x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
		if self.sub is not None: x.sub_(self.sub)
		if self.maxv is not None: x.clamp_max_(self.maxv)
		return x
