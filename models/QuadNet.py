import torch
import numpy as np
import torch.nn as nn

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(
                -np.sqrt(6 / num_input) / 30,
                np.sqrt(6 / num_input) / 30
            )


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Sine(nn.Module):
    def __init__(self, const=15.):
        super().__init__()
        self.const = const

    def forward(self, x):
        return torch.sin(self.const * x)


# -------------------- Hybrid SIREN + Quadratic Net --------------------

class Net(nn.Module):
    """
    SIREN with explicit quadratic skip:
        f(x) = f_siren(x) + x^T A(x) x
    """

    def __init__(self, _, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.out_dim = cfg.out_dim
        self.hidden_size = cfg.hidden_size
        self.n_blocks = cfg.n_blocks

        # ---------------- SIREN backbone ----------------
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Linear(self.dim, self.hidden_size))
        for _ in range(self.n_blocks):
            self.blocks.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.blocks.append(nn.Linear(self.hidden_size, self.out_dim))
        if self.dim == 2:
            print("test")
            self.act = Sine(10.)
        else:
            self.act = Sine(25.)
        # ---------------- Quadratic head ----------------
        # outputs dim * dim coefficients per point
        self.quad_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            Sine(),
            nn.Linear(self.hidden_size, self.dim * self.dim)
        )

        # ---------------- Initialization ----------------
        self.apply(sine_init)
        self.blocks[0].apply(first_layer_sine_init)
        self.quad_head.apply(sine_init)

        if getattr(cfg, "zero_init_last_layer", False):
            nn.init.constant_(self.blocks[-1].weight, 0)
            nn.init.constant_(self.blocks[-1].bias, 0)

    def forward(self, x, phi=0):
        """
        x: (bs, npoints, dim)
        return: (bs, npoints, out_dim)
        """

        net = x
        for block in self.blocks[:-1]:
            net = self.act(block(net))

        siren_out = self.blocks[-1](net)

        # -------- quadratic correction --------
        # A(x): (bs, npoints, dim, dim)
        A = self.quad_head(net).view(
            *net.shape[:-1], self.dim, self.dim
        )

        quad = torch.einsum('...i,...ij,...j->...', x, A, x)
        
        quad = quad.unsqueeze(-1)
        return siren_out + quad
