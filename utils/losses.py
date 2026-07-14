import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#%%
class ContrastiveLoss(nn.Module):

    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def _info_nce(self, z, zp):
        """
        Args:
            z:  (B, L, D)
            zp: (B, L, D)

        Returns:
            scalar loss
        """
        z = z.mean(dim=1)      # (B, D)
        zp = zp.mean(dim=1)    # (B, D)
        z = F.normalize(z, dim=-1)
        zp = F.normalize(zp, dim=-1)
        sim = torch.matmul(z, zp.transpose(0, 1)) / self.tau           # (B, B)
        labels = torch.arange(z.size(0), device=z.device)
        loss = F.cross_entropy(sim,labels)

        return loss

    def forward(self, z, z_list):
        """
        Args:
            z:      (B, L, D)
            z_list: list[(B, L, D)]

        Returns:
            scalar loss
        """

        loss = 0.
        for zp in z_list:
            loss += self._info_nce(z, zp)
        loss /= len(z_list)

        return loss