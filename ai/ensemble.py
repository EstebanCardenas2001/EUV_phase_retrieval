import torch
import torch.nn as nn

class Ensemble(nn.Module):
    """
    Deep ensemble of independently trained phase-retrieval models (same OpticsConfig, different seeds).
    forward() returns the member-mean phase (and coefficients), matching the single-model interface.
    """
    def __init__(self, models):
        super().__init__()
        self.members = nn.ModuleList(models)

    def forward(self, x: torch.Tensor, return_coeffs: bool = False):
        outs = [m(x, return_coeffs=True) for m in self.members]
        phase = torch.stack([o[0] for o in outs]).mean(0)
        if return_coeffs:
            return phase, torch.stack([o[1] for o in outs]).mean(0)
        return phase

    def enable_mc_dropout(self):
        for m in self.members:
            m.enable_mc_dropout()
