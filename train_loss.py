from itertools import count
import torch.nn as nn
import torch
import torch.nn.functional as F
bce = nn.BCELoss(reduction='mean')


def multi_edge(pred, edge_p, gt, edge):
    loss=0.
    for i in range(len(pred)):
        if i == 0:
            loss+= (bce(pred[i], gt)*(3**3/30) + bce(edge_p[i], edge)*(3**3/30/2)) 
        elif i==1:
            loss+= (bce(pred[i], gt)*(3**2/30) + bce(edge_p[i], edge)*(3**2/30/2))
        else:
            loss+= (bce(pred[i], gt)*(3**1/30) + bce(edge_p[i], edge)*(3**1/30/2))
    return loss


