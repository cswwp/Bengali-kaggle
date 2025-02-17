import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class FocalLoss(nn.Module):
   def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
       super(FocalLoss, self).__init__()
       if alpha is None:
           self.alpha = Variable(torch.ones(class_num, 1))
       else:
           if isinstance(alpha, Variable):
               self.alpha = alpha
           else:
               self.alpha = Variable(alpha)
       self.alpha = self.alpha.cuda()
       self.gamma = gamma
       self.class_num = class_num
       self.size_average = size_average
   def forward(self, inputs, targets):
       N = inputs.size(0)
       C = inputs.size(1)
       P = F.softmax(inputs)
       class_mask = inputs.data.new(N, C).fill_(0)
       class_mask = Variable(class_mask)
       ids = targets.view(-1, 1)
       class_mask.scatter_(1, ids.data, 1.)
       #print(class_mask.shape)
       #print(P.shape)

       # if inputs.is_cuda and not self.alpha.is_cuda:
       #     self.alpha = self.alpha.to(device)
       alpha = self.alpha[ids.data.view(-1)]
       probs = (P*class_mask).sum(1).view(-1,1)
       log_p = probs.log()
       #print('probs size= {}'.format(probs.size()))
      # print(probs)
       batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
       #print('-----bacth_loss------')
       #print(batch_loss)
       if self.size_average:
           loss = batch_loss.mean()
       else:
           loss = batch_loss.sum()
       return loss

if __name__ == '__main__':
    loss = FocalLoss(16)
    img = torch.rand(3, 16).cuda()
    label = torch.tensor([0, 4, 10]).cuda()
    res = loss(img, label)

    print('rest:', res)
