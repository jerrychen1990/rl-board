#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     model.py
   Author :       chenhao
   time：          2022/2/15 14:55
   Description :
-------------------------------------------------
"""
from abc import abstractmethod
from functools import reduce
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, Tanh, Conv2d, Flatten
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from rlb.actor_critic import ModuleActorCritic
from rlb.core import Step, Piece
from rlb.gravity_gobang import GravityGoBang3
from rlb.tictactoe import TicTacToe


class MLPActorCritic(ModuleActorCritic):
    def __init__(self, dims: List, action_num, *args, **kwargs):
        super(MLPActorCritic, self).__init__(action_num=action_num, *args, **kwargs)
        fc_layers = []
        in_dim = reduce(lambda x, y: x * y, self.input_shape)
        for dim in dims:
            fc_layers.append(Linear(in_dim, dim))
            fc_layers.append(ReLU())
            in_dim = dim
        self.encoder = Sequential(*fc_layers)
        self.critic_decoder = Linear(in_dim, 1)
        self.actor_decoder = Linear(in_dim, self.action_num)

    @abstractmethod
    def obs2tensor(self, obs) -> Tensor:
        raise NotImplementedError

    def forward(self, x):
        features = self.encoder(x)
        value = self.critic_decoder(features)
        value = Tanh()(value).squeeze(-1)
        weights = self.actor_decoder(features)
        return weights, value

    def act_and_criticize(self, obs) -> Tuple[List[float], float]:
        x = self.obs2tensor(obs)
        weights, value = self.forward(x)
        probs, value = F.softmax(weights, dim=-1).detach().numpy(), value.item()
        return probs, value

    def criticize(self, obs) -> float:
        x = self.obs2tensor(obs)
        features = self.encoder(x)
        value = self.critic_decoder(features).item()
        return value

    def act(self, obs) -> List[float]:
        x = self.obs2tensor(obs)
        features = self.encoder(x)
        weights = self.actor_decoder(features)
        probs = F.softmax(weights).detach().numpy()
        return probs

    @property
    @abstractmethod
    def input_shape(self):
        raise NotImplementedError


class CNNActorCritic(ModuleActorCritic):
    def __init__(self, kernels: List, state_shape, action_num, *args, **kwargs):
        super(CNNActorCritic, self).__init__(*args, **kwargs)
        self.action_num = action_num
        self.state_shape = state_shape
        in_channels = self.state_shape[0]

        conv_layers = []

        for out_channels, kernel_size in kernels:
            conv_layer = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
            conv_layers.append(conv_layer)
            in_channels = out_channels
            conv_layers.append(ReLU())
        self.encoder = Sequential(*conv_layers, Flatten())

        in_dim = self._get_convs_output_dim()
        self.critic_decoder = Linear(in_dim, 1)
        self.actor_decoder = Linear(in_dim, self.action_num)

    def _get_convs_output_dim(self):
        return self.encoder(torch.zeros(1, *self.state_shape)).view(1, -1).size(1)

    @abstractmethod
    def obs2tensor(self, obs) -> Tensor:
        raise NotImplementedError

    def forward(self, x):
        features = self.encoder(x)
        value = self.critic_decoder(features)
        value = Tanh()(value).squeeze(-1)
        weights = self.actor_decoder(features)
        return weights, value

    def act_and_criticize(self, obs) -> Tuple[List[float], float]:
        x = self.obs2tensor(obs)
        weights, value = self.forward(x)
        probs, value = F.softmax(weights).detach().numpy(), value.item()
        return probs, value

    def criticize(self, obs) -> float:
        x = self.obs2tensor(obs)
        features = self.encoder(x)
        value = self.critic_decoder(features).item()
        return value

    def act(self, obs) -> List[float]:
        x = self.obs2tensor(obs)
        features = self.encoder(x)
        weights = self.actor_decoder(features)
        probs = F.softmax(weights).detach().numpy()
        return probs


def eval_ac_model(ac_model, steps: List[Step]):
    obs = torch.stack([ac_model.obs2tensor(s.obs) for s in steps])
    tgt_probs = torch.from_numpy(np.array([s.probs for s in steps])).float()
    tgt_values = torch.from_numpy(np.array([s.extra_info["value"] for s in steps])).float()
    with torch.no_grad():
        weights, values = ac_model.forward(obs)
        probs = F.softmax(weights, dim=-1).detach()
        actor_loss = F.cross_entropy(probs, tgt_probs).item()
        critic_loss = F.mse_loss(values, tgt_values).item()
        loss = actor_loss + critic_loss
        acc = ((tgt_probs > 0).int() * probs).sum(axis=-1).mean().detach().item()
        return dict(acc=acc, loss=loss, actor_loss=actor_loss, critic_loss=critic_loss)


class BoardMLPActorCritic(MLPActorCritic):
    def __init__(self, board_size, *args, **kwargs):
        self.board_size = board_size
        super(BoardMLPActorCritic, self).__init__(*args, **kwargs)


    def obs2tensor(self, obs) -> Tensor:
        t = []
        b, p = obs
        for row in b:
            for e in row:
                if e == Piece.BLANK:
                    t.append(0)
                elif e == p:
                    t.append(1)
                else:
                    t.append(-1)
        t = torch.tensor(t).float()
        return t

    @property
    def input_shape(self):
        return self.board_size, self.board_size


class TicTacToeConvActorCritic(CNNActorCritic):
    def obs2tensor(self, obs) -> Tensor:
        t = []
        b, p = obs
        for row in b:
            tmp = []
            for e in row:
                if e == Piece.BLANK:
                    tmp.append([1, 0, 0])
                elif e == p:
                    tmp.append([0, 1, 0])
                else:
                    tmp.append([0, 0, 1])
            t.append(tmp)
        t = torch.tensor(t).float().permute(2, 0, 1)
        return t


def get_model_cls(env_cls, model_type):
    _ac_model_dict = {
        (TicTacToe, "MLP"): BoardMLPActorCritic,
        (GravityGoBang3, "MLP"): BoardMLPActorCritic
    }
    key = env_cls, model_type
    if key not in _ac_model_dict:
        raise Exception(f"invalid key:{key}, valid keys:{_ac_model_dict.keys()}")
    return _ac_model_dict[key]


def get_optimizer_cls(optimizer_name):
    _opt_model_dict = {
        "ADAM": Adam
    }
    key = optimizer_name.upper()
    if key not in _opt_model_dict:
        raise Exception(f"invalid key:{key}, valid keys:{_opt_model_dict.keys()}")
    return _opt_model_dict[key]


def get_schedule_cls(schedule_name):
    _schedule_model_dict = {
        "StepLR".upper(): StepLR,
        "ExponentialLR".upper(): ExponentialLR
    }
    key = schedule_name.upper()
    if key not in _schedule_model_dict:
        raise Exception(f"invalid key:{key}, valid keys:{_schedule_model_dict.keys()}")
    return _schedule_model_dict[key]
