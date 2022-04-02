#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     core.py
   Author :       chenhao
   time：          2022/2/15 10:34
   Description :
-------------------------------------------------
"""
from __future__ import annotations

import logging
import os
from abc import abstractmethod
from multiprocessing import Process
from typing import Tuple, List, Type

from pydantic import BaseModel

from rlb.utils import sample_by_probs

logger = logging.getLogger(__name__)


class Action(BaseModel):
    is_pass: bool = False

    @abstractmethod
    def to_idx(self) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_idx(cls, idx: int) -> Action:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_cmd(cls, cmd: str) -> Action:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_pass_action(cls) -> Action:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        pass

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, Action) and hash(self) == hash(other)


class State(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def to_tuple(self) -> Tuple:
        raise NotImplementedError

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        return type(other) == type(self) and hash(self) == hash(other)

    @abstractmethod
    def render_str(self) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_render_str(cls, render_str: str):
        raise NotImplementedError

    def __str__(self):
        return self.render_str()


class TransferInfo(BaseModel):
    next_state: State
    reward: float
    is_done: bool
    extra_info: dict


class Env:
    action_cls: Type[Action] = NotImplemented
    action_num: int = NotImplemented

    @abstractmethod
    def get_state(self) -> State:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def transfer(cls, state: State, action: Action) -> TransferInfo:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> TransferInfo:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_valid_actions_by_state(cls, state: State) -> List[Action]:
        raise NotImplementedError

    def get_valid_actions(self) -> List[Action]:
        state = self.get_state()
        return self.get_valid_actions_by_state(state=state)

    @abstractmethod
    def reset(self) -> State:
        raise NotImplementedError

    @abstractmethod
    def render(self, mode="human"):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed):
        raise NotImplementedError


class ActionInfo(BaseModel):
    action_idx: int
    prob: float
    probs: List[float]


class Agent:
    def __init__(self, name, action_num, *args, **kwargs):
        self.name = name
        self.action_num = action_num

    def get_weights(self, state: State, mode: str, **kwargs) -> List[float]:
        raise NotImplementedError

    def choose_action(self, state: State, mode: str, **kwargs) -> ActionInfo:
        weights = self.get_weights(state=state, mode=mode, **kwargs)
        assert min(weights) >= 0
        sum_weights = sum(weights)
        if sum_weights == 0:
            probs = [1 / self.action_num] * self.action_num
        else:
            probs = [w / sum_weights for w in weights]
        action_idx, prob = sample_by_probs(probs=probs)
        # logger.debug(f"choose action:{action_idx} with prob:{prob:2.3f}")
        action_info = ActionInfo(action_idx=action_idx, prob=prob, probs=probs)
        return action_info

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def on_episode_end(self, episode_idx: int):
        pass


class RandomAgent(Agent):
    def get_weights(self, **kwargs) -> List[float]:
        return [1 / self.action_num] * self.action_num


class Context(BaseModel):
    base_dir: str

    @property
    def model_dir(self):
        return os.path.join(self.base_dir, "models")

    @property
    def best_model_path(self):
        return os.path.join(self.model_dir, "best_model.pt")

    @property
    def record_dir(self):
        return os.path.join(self.base_dir, "records")

    def ckpt_model_path(self, ckpt: int):
        return os.path.join(self.model_dir, f"model-{ckpt}.pt")

    @property
    def config_path(self):
        return os.path.join(self.base_dir, "config.json")



class BaseProcess(Process):
    def __init__(self, context: Context, run_kwargs=dict(), *args, **kwargs):
        super(BaseProcess, self).__init__(*args, **kwargs)
        self.context = context
        self.run_kwargs = run_kwargs
