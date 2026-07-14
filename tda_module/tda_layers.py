import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologicalPotentialBank(nn.Module):

    def __init__(
        self,
        n_kernels=32,
        eps=1e-3,
        pooling="max"
    ):
        super().__init__()

        self.n_kernels = n_kernels
        self.eps = eps
        self.pooling = pooling

        # ==================================================
        # Learnable Centers
        # ==================================================

        self.mu_x = nn.Parameter(
            torch.rand(n_kernels)
        )

        self.mu_y = nn.Parameter(
            torch.rand(n_kernels)
        )

        # ==================================================
        # Learnable Widths
        # ==================================================

        self.log_sigma_x = nn.Parameter(
            torch.zeros(n_kernels)
        )

        self.log_sigma_y = nn.Parameter(
            torch.zeros(n_kernels)
        )

        # ==================================================
        # Oscillatory Parameters
        # ==================================================

        self.alpha = nn.Parameter(
            torch.ones(n_kernels)
        )

        self.beta_raw = nn.Parameter(
            torch.zeros(n_kernels)
        )

        self.omega = nn.Parameter(
            torch.ones(n_kernels)
        )
    
    def kernel_response(self, x, y):
        """
        Parameters
        ----------
        x : (..., 1)
        y : (..., 1)
    
        Returns
        -------
        phi : (..., N)
        """
    
        sigma_x = F.softplus(self.log_sigma_x) + 1e-6
        sigma_y = F.softplus(self.log_sigma_y) + 1e-6
    
        alpha = F.softplus(self.alpha)
        beta = torch.sigmoid(self.beta_raw)
        omega = F.softplus(self.omega)
    
        r = torch.sqrt(
            self.eps ** 2
            + ((x - self.mu_x) / sigma_x) ** 2
            + ((y - self.mu_y) / sigma_y) ** 2
        )
    
        phi = (
            y
            * torch.exp(-alpha * r)
            * (
                1.0
                + beta * torch.sin(omega * r)
            )
        )
    
        return phi

    def forward(
        self,
        dgms,
        mask=None):
        """
        dgms : (B,M,2)
        mask : (B,M)
        """
    
        x = dgms[..., 0].unsqueeze(-1)
        y = dgms[..., 1].unsqueeze(-1)
    
        # ==================================================
        # Kernel responses
        # ==================================================
    
        phi = self.kernel_response(x, y)
    
        # (B,M,N)
    
        # ==================================================
        # Apply mask
        # ==================================================
    
        if mask is not None:
            phi = phi * mask.unsqueeze(-1)
    
        # ==================================================
        # Pooling
        # ==================================================
    
        if self.pooling == "sum":
    
            features = phi.sum(dim=1)
    
        elif self.pooling == "mean":
    
            if mask is None:
    
                features = phi.mean(dim=1)
    
            else:
    
                denom = (
                    mask.sum(dim=1, keepdim=True)
                    + 1e-8
                )
    
                features = (
                    phi.sum(dim=1)
                    / denom
                )
    
        elif self.pooling == "max":
    
            features = phi.max(dim=1)[0]
    
        else:
    
            raise ValueError(
                f"Unknown pooling: {self.pooling}"
            )
        
        return {
            "features": features,
            "responses": phi
        }