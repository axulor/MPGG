import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .util import init, get_clones
import argparse
from typing import List, Tuple, Union, Optional

class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        layer_N: int,
        use_orthogonal: bool,
        use_ReLU: bool,
    ):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            active_func,
            nn.LayerNorm(hidden_size),
        )
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        obs_shape: Union[List, Tuple],
        override_obs_dim: Optional[int] = None,
    ):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        if override_obs_dim is None:
            obs_dim = obs_shape[0]
        else:
            print("Overriding Observation dimension")
            obs_dim = override_obs_dim

        self.obs_dim = obs_dim

        if self._use_feature_normalization:
            print(f"[DEBUG] LayerNorm expects normalized_shape = {obs_dim}")
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim,
            self.hidden_size,
            self._layer_N,
            self._use_orthogonal,
            self._use_ReLU,
        )

    def forward(self, x):
        print(f"[DEBUG] MLP forward input shape = {x.shape}")
        if self._use_feature_normalization:
            if x.shape[-1] < self.obs_dim:
                pad_size = self.obs_dim - x.shape[-1]
                pad = torch.zeros(x.shape[0], pad_size, device=x.device)
                x = torch.cat([x, pad], dim=-1)
                print(f"[DEBUG] Padded input to shape: {x.shape}")
            elif x.shape[-1] > self.obs_dim:
                x = x[..., :self.obs_dim]
                print(f"[DEBUG] Truncated input to shape: {x.shape}")
            x = self.feature_norm(x)

        return self.mlp(x)