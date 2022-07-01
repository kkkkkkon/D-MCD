import numpy as np
import torch
import sklearn
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
# from numpy.random import *
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch
from torch.optim.optimizer import Optimizer, required

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss

def draw_data_TSNE(src_images, src_label, tgt_images, tgt_label, name=''):
    X1 = TSNE(n_components=2).fit_transform(src_images.cpu().numpy())
    Y1 = src_label.cpu().numpy()
    X2 = TSNE(n_components=2).fit_transform(tgt_images.cpu().numpy())
    Y2 = tgt_label.cpu().numpy()
    plt.cla()
    plt.scatter(X1[:, 0], X1[:, 1], 10, c='r')
    plt.scatter(X2[:, 0], X2[:, 1], 10, c='b')
    plt.savefig('train_process/vision/img' + name + '.jpg')
    return
 
def draw_data(data, label, name=''):
    X1 = TSNE(n_components=2).fit_transform(data.cpu().numpy())
    Y1 = label.cpu().numpy()
    plt.cla()
    plt.scatter(X1[:, 0], X1[:, 1], 10, label)
    plt.savefig('train_process/vision/img' + name + '.jpg')
    return

def soft_criterion_weight(s_pre, t_pre, w):
    batch_size, dim = s_pre.size()
    return -torch.sum(torch.mul(t_pre, torch.log(s_pre + 1e-30)) * w) / (batch_size)

def soft_criterion(s_pre, t_pre):
    batch_size, dim = s_pre.size()
    return -torch.sum(torch.mul(t_pre, torch.log(s_pre + 1e-4))) / (batch_size )

def textread(path):
    f = open(path)
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        lines[i] = lines[i].replace('\n', '')
    return lines

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def adjust_ensemble_rate(ratio, beta=15, gamma=-1.75):
    rate = (beta * ratio) ** (gamma)
    return rate
    
def adjust_learning_rate(optimizer, epoch,iter_num,iter_per_epoch=4762,lr=0.001):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    ratio=(epoch*iter_per_epoch+iter_num)/(10*iter_per_epoch)
    lr=lr*(1+10*ratio)**(-0.75)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        # m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)

def cdd(output_t1, output_t2):
    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = torch.sum(mul) - torch.trace(mul)
    return cdd_loss 

def get_theta(embedding_dim, num_samples=50):
    theta = [w/np.sqrt((w**2).sum())
             for w in np.random.normal(size=(num_samples, embedding_dim))]
    theta = np.asarray(theta)
    return torch.from_numpy(theta).type(torch.FloatTensor).cuda()

def sliced_wasserstein_distance(source_z, target_z,embed_dim, num_projections=256, p=1):
    # theta is vector represents the projection directoin
    batch_size = target_z.size(0)
    theta = get_theta(embed_dim, num_projections)
    proj_target = target_z.matmul(theta.transpose(0, 1))
    proj_source = source_z.matmul(theta.transpose(0, 1))

    w_distance = torch.abs(torch.sort(proj_target.transpose(0, 1), dim=1)[0]-torch.sort(proj_source.transpose(0, 1), dim=1)[0])
    w_distance=torch.mean(w_distance)
    # calculate by the definition of p-Wasserstein distance
    w_distance_p = torch.pow(w_distance, p)

    return w_distance_p.mean()




