import torch
from torch import nn


class GR(nn.Module):
    def __init__(self, weight_loss_GR, sinkhorn_alpha, OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_GR = weight_loss_GR
        self.stopThr = stopThr
        self.epsilon = 1e-16
        self.transp = None

    def forward(self, M, group):
        if self.weight_loss_GR <= 1e-6:
            return 0.
        else:
            # M: KxV cost matrix
            # sinkhorn alpha = 1/ entropic reg weight
            # a: Kx1 source distribution
            # b: Vx1 target distribution
            device = M.device
            group = group.to(device)

            # Sinkhorn's algorithm
            a = (group.sum(axis=1)).unsqueeze(1).to(device)
            b = (group.sum(axis=0)).unsqueeze(1).to(device)

            u = (torch.ones_like(a) / a.size()[0]).to(device) # Kx1

            K = torch.exp(-M * self.sinkhorn_alpha)
            err = 1
            cpt = 0
            while err > self.stopThr and cpt < self.OT_max_iter:
                v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
                u = torch.div(a, torch.matmul(K, v) + self.epsilon)
                cpt += 1
                if cpt % 50 == 1:
                    bb = torch.mul(v, torch.matmul(K.t(), u))
                    err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

            transp = u * (K * v.T)
            transp = transp.clamp(min=1e-6)

            self.transp = transp

            loss_GR = (group * (group.log() - transp.log() - 1) \
                + transp).sum()
            loss_GR *= self.weight_loss_GR

            return loss_GR
