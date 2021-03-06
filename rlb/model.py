# ! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     model.py
   Author :       chenhao
   time：          2022/2/15 14:55
   Description :
-------------------------------------------------
"""

import logging
import math
from abc import abstractmethod, ABC
from functools import reduce, lru_cache
from typing import List, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from snippets import get_batched_data
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU, Tanh, Conv2d, Flatten, Module, MaxPool2d
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from rlb.actor_critic import ActorCritic
from rlb.core import Piece, BoardEnv, ACInfo, State
from rlb.core import Context

logger = logging.getLogger(__name__)


class ModuleActorCritic(Module, ActorCritic):
    def __init__(self, action_num, *args, **kwargs):
        Module.__init__(self, *args, **kwargs)
        ActorCritic.__init__(self, action_num=action_num)

    def ac_info2arrays(self, ac_infos: List[ACInfo]):
        act_obs = np.array([self.state2array(s.state) for s in ac_infos if s.probs is not None]).astype(np.float32)
        tgt_probs = np.array([s.probs for s in ac_infos if s.probs is not None]).astype(np.float32)

        critic_obs = np.array([self.state2array(s.state) for s in ac_infos]).astype(np.float32)
        tgt_values = np.array([s.value for s in ac_infos]).astype(np.float32)
        return act_obs, critic_obs, tgt_probs, tgt_values

    def learn_on_batch(self, ac_infos: List[ACInfo], optimizer: Optimizer, actor_loss_type="ce"):
        act_obs, critic_obs, tgt_probs, tgt_values = [torch.from_numpy(e) for e in self.ac_info2arrays(ac_infos)]
        weights = self.forward_act(act_obs)
        values = self.forward_critic(critic_obs)
        if actor_loss_type == "ce":
            actor_loss = F.cross_entropy(weights, tgt_probs)
        elif actor_loss_type == "kl":
            log_probs = F.log_softmax(weights, dim=-1)
            actor_loss = F.kl_div(log_probs, tgt_probs)
        else:
            raise ValueError(f"invalid {actor_loss_type=}")

        value_loss = F.mse_loss(values, tgt_values)

        loss = actor_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():  # clip防止梯度爆炸
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return weights, tgt_probs, values, tgt_values, loss

    def train(self, ac_infos: List[ACInfo], optimizer: Optimizer, epochs: int, mini_batch_size=32,
              schedule=None, show_lines=None):
        logger.info(f"training model on {len(ac_infos)} ac_infos")
        interval = int(math.ceil(epochs / show_lines)) if show_lines else 1

        for epoch in range(epochs):
            losses = []

            for batch in get_batched_data(ac_infos, mini_batch_size):
                losses.append(self.learn_on_batch(ac_infos=batch, optimizer=optimizer)[-1].detach().item())
            loss = sum(losses) / len(losses)
            if (epoch + 1) % interval == 0:
                schedule_info = f"[lr:{schedule.get_last_lr()[0]:1.6f}]" if schedule else ""
                logger.info(f"[{epoch + 1}/{epochs}] {schedule_info} {loss=:2.3f}")
            if schedule:
                schedule.step()

    @abstractmethod
    def forward_act(self, x):
        raise NotImplementedError

    @abstractmethod
    def forward_critic(self, x):
        raise NotImplementedError

    def state2tensor(self, state: State) -> Tensor:
        return torch.from_numpy(self.state2array(state)).float()

    @abstractmethod
    def state2array(self, state: State) -> np.array:
        raise NotImplementedError

    @lru_cache(maxsize=5000)
    def act_and_criticize(self, state: State) -> Tuple[List[float], float]:
        x = self.state2tensor(state)
        weights, value = self.forward(x)
        probs, value = F.softmax(weights, dim=-1).detach().numpy(), value.item()
        return probs, value

    @lru_cache(maxsize=5000)
    def criticize(self, state: State) -> float:
        x = self.state2tensor(state)
        features = self.encoder(x)
        value = self.critic_decoder(features).item()
        return value

    @lru_cache(maxsize=5000)
    def act(self, state) -> List[float]:
        x = self.state2tensor(state)
        features = self.encoder(x)
        weights = self.actor_decoder(features)
        probs = F.softmax(weights).detach().numpy()
        return probs

    def clean_cache(self):
        self.act.cache_clear()
        self.criticize.cache_clear()
        self.act_and_criticize.cache_clear()

    @abstractmethod
    def input_shape(self):
        raise NotImplementedError


class MLPActorCritic(ModuleActorCritic, ABC):
    def __init__(self, dims: List, action_num, *args, **kwargs):
        super(MLPActorCritic, self).__init__(action_num=action_num, *args, **kwargs)
        fc_layers = []
        in_dim = reduce(lambda x, y: x * y, self.input_shape())
        for dim in dims:
            fc_layers.append(Linear(in_dim, dim))
            fc_layers.append(ReLU())
            in_dim = dim
        self.encoder = Sequential(*fc_layers)
        self.critic_decoder = Linear(in_dim, 1)
        self.actor_decoder = Linear(in_dim, self.action_num)

    def forward(self, x):
        features = self.encoder(x)
        value = self.critic_decoder(features)
        value = Tanh()(value).squeeze(-1)
        weights = self.actor_decoder(features)
        return weights, value

    def forward_act(self, x):
        features = self.encoder(x)
        weights = self.actor_decoder(features)
        return weights

    def forward_critic(self, x):
        features = self.encoder(x)
        values = self.critic_decoder(features)
        values = Tanh()(values).squeeze(-1)
        return values


class CNNActorCritic(ModuleActorCritic, ABC):
    def __init__(self, kernels: List, action_num, *args, **kwargs):
        super(CNNActorCritic, self).__init__(action_num=action_num, *args, **kwargs)
        in_channels = self.state_shape[0]

        conv_layers = []

        for out_channels, kernel_size, max_pool_size in kernels:
            conv_layer = Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding="same")
            conv_layers.append(conv_layer)
            in_channels = out_channels
            conv_layers.append(ReLU())
            if max_pool_size:
                conv_layers.append(MaxPool2d(kernel_size))

        self.encoder = Sequential(*conv_layers, Flatten())

        in_dim = self._get_convs_output_dim()
        self.critic_decoder = Linear(in_dim, 1)
        self.actor_decoder = Linear(in_dim, self.action_num)

    def _get_convs_output_dim(self):
        logger.info(self.state_shape)
        rs = self.encoder(torch.zeros(1, *self.state_shape)).view(1, -1).size(1)
        logger.info(rs)
        return rs

    @property
    @abstractmethod
    def state_shape(self):
        raise NotImplementedError

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        features = self.encoder(x)
        values = self.critic_decoder(features)
        values = Tanh()(values).squeeze(-1)
        weights = self.actor_decoder(features)
        return weights, values

    def forward_act(self, x):
        features = self.encoder(x)
        weights = self.actor_decoder(features)
        return weights

    def forward_critic(self, x):
        features = self.encoder(x)
        values = self.critic_decoder(features)
        values = Tanh()(values).squeeze(-1)
        return values


class BoardMLPActorCritic(MLPActorCritic):
    def __init__(self, board_size, *args, **kwargs):
        self.board_size = board_size
        super(BoardMLPActorCritic, self).__init__(*args, **kwargs)

    def state2array(self, state: State) -> np.array:
        t = []
        for row in state.get_board().gen_rows():
            for e in row:
                if e == Piece.B:
                    t.append(0)
                elif e == state.piece:
                    t.append(1)
                else:
                    t.append(-1)
        t = np.array(t)
        return t

    def input_shape(self):
        return self.board_size * self.board_size,


class BoardCNNActorCritic(CNNActorCritic, ABC):
    def __init__(self, board_size, *args, **kwargs):
        self.board_size = board_size
        super(BoardCNNActorCritic, self).__init__(*args, **kwargs)

    @property
    def state_shape(self):
        return 3, self.board_size, self.board_size

    def state2tensor(self, state) -> Tensor:
        t = []
        b, p = state
        for row in b:
            tmp = []
            for e in row:
                if e == Piece.B:
                    tmp.append([1, 0, 0])
                elif e == p:
                    tmp.append([0, 1, 0])
                else:
                    tmp.append([0, 0, 1])
            t.append(tmp)
        t = np.array(t).transpose([2, 0, 1])
        return t


def get_model_cls(model_type: str):
    _ac_model_dict = {
        "MLP": BoardMLPActorCritic,
        "CNN": BoardCNNActorCritic
    }
    if model_type in _ac_model_dict:
        return _ac_model_dict[model_type]
    raise Exception(f"invalid model_type:{model_type}, valid keys:{_ac_model_dict.keys()}")


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


def load_ac_model(context: Context, ckpt: int):
    if ckpt == -1:
        model_path = context.best_model_path
    else:
        model_path = context.ckpt_model_path(ckpt=ckpt)
    logger.info(f"loading ac model from path:{model_path}")
    ac_model = torch.load(model_path)
    return ac_model


def build_ac_model(env: BoardEnv, model_type: str, torch_kwargs: dict):
    ac_model_cls = get_model_cls(model_type)
    ac_model = ac_model_cls(**torch_kwargs, action_num=env.action_num, board_size=env.board_size)
    return ac_model
