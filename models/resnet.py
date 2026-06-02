import torch
import torch.nn as nn

# ------------------
# Activation
# ------------------
class Sine(nn.Module):
    def __init__(self, w=10.0):
        super().__init__()
        self.w = w

    def forward(self, x):
        return torch.sin(self.w * x)


# ------------------
# Residual Block
# ------------------
class ResidualBlock(nn.Module):
    def __init__(self, width, activation=nn.ReLU(), alpha=0.5):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.alpha = alpha

        # SIREN-style initialization
        if isinstance(activation, Sine):
            w0 = activation.w
            fan_in = self.fc1.weight.size(1)
            nn.init.uniform_(self.fc1.weight, - (6 / fan_in)**0.5 / w0, (6 / fan_in)**0.5 / w0)
            nn.init.uniform_(self.fc2.weight, - (6 / fan_in)**0.5 / w0, (6 / fan_in)**0.5 / w0)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)


    def forward(self, x):
        out = self.activation(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return x + self.alpha * out


# ------------------
# Network
# ------------------
class Net(nn.Module):
    def __init__(self, _, cfg):
        super().__init__()
        self.in_dim = cfg.dim
        if self.in_dim == 2:
            activation = Sine(w=25.0)
        else:
            activation = Sine(w=5.0)

        self.input_layer = nn.Linear(cfg.dim, cfg.hidden_size)
        # SIREN init for input layer
        if isinstance(activation, Sine):
            fan_in = cfg.dim
            w0 = activation.w
            nn.init.uniform_(self.input_layer.weight, - (1 / fan_in), (1 / fan_in))
            nn.init.zeros_(self.input_layer.bias)
        if (self.in_dim == 2):
            self.res_blocks = nn.Sequential(
                *[ResidualBlock(cfg.hidden_size, activation, alpha=1.) for _ in range(cfg.n_blocks)]
            )
        else:
            self.res_blocks = nn.Sequential(
                *[ResidualBlock(cfg.hidden_size, activation, alpha=0.5) for _ in range(cfg.n_blocks)]
            )

        self.output_layer = nn.Linear(cfg.hidden_size, cfg.out_dim)
        nn.init.zeros_(self.output_layer.weight)
        b = 1.0 + torch.empty(1).uniform_(-0.01, 0.01)
        nn.init.constant_(self.output_layer.bias, b.item())

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        if self.in_dim == 2:
            f = nn.functional.sigmoid(x/10)
        else:
            f = nn.functional.sigmoid(x/10)
        x = f
        return x