

#%%
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

#%%


class CorrLoss(nn.Module):
    def __init__(self, eps=1e-12, detach_target=True):
        """
        Compute correlation between the output and the target
        Args:
            eps (float, optional): Used to offset the computed variance to provide numerical stability.
                Defaults to 1e-12.
            detach_target (bool, optional): If True, `target` tensor is detached prior to computation. Appropriate when
                using this as a loss to train on. Defaults to True.
        """
        self.eps = eps
        self.detach_target = detach_target
        super().__init__()

    def forward(self, output, target, mask=None):
        if self.detach_target:
            target = target.detach()
        delta_out = output - output.mean(0, keepdim=True)
        delta_target = target - target.mean(0, keepdim=True)

        var_out = delta_out.pow(2).mean(0, keepdim=True)
        var_target = delta_target.pow(2).mean(0, keepdim=True)

        corrs = (delta_out * delta_target).mean(0, keepdim=True) / (
            (var_out + self.eps) * (var_target + self.eps)
        ).sqrt()

        return corrs

#%%
n = 1000
x = torch.randn((n,1)).requires_grad_()

y = 5*x + 1 + torch.randn((n,1))

closs = CorrLoss()

print(closs(x, y))


plt.plot(x.detach(), y.detach(), '.')


# %%
