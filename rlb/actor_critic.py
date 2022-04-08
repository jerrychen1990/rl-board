#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     actor_critic.py
   Author :       chenhao
   time：          2022/2/18 14:19
   Description :
-------------------------------------------------
"""
import logging
import random
from abc import ABCMeta
from typing import List, Tuple, Type

import numpy as np

from rlb.core import State, Agent, BoardEnv
from rlb.utils import weights2probs


class Critic:
    def criticize(self, state: State) -> float:
        raise NotImplementedError


class Actor:
    def __init__(self, action_num):
        self.action_num = action_num

    def act(self, state: State) -> List[float]:
        raise NotImplementedError


class ActorCritic(Critic, Actor, metaclass=ABCMeta):
    def act_and_criticize(self, state: State) -> Tuple[List[float], float]:
        raise NotImplementedError


class RandomActorCritic(ActorCritic):

    def criticize(self, state: State) -> float:
        return random.uniform(-1, 1)

    def act(self, state: State) -> List[float]:
        weights = np.array([random.random() for _ in range(len(self.action_num))])
        probs = weights2probs(weights)
        return probs

    def act_and_criticize(self, state: State) -> Tuple[List[float], float]:
        v = self.criticize(state)
        probs = self.act(state)
        return probs, v


class ActorCriticAgent(Agent):
    def __init__(self, ac: ActorCritic, env: BoardEnv, *args, **kwargs):
        assert ac.action_num == self.env.action_num
        super(ActorCriticAgent, self).__init__(action_num=ac.action_num, *args, **kwargs)
        self.ac = ac
        self.env = env

    def get_weights(self, state, mode, **kwargs) -> List[float]:
        logging.debug(state)
        probs, value = self.ac.act_and_criticize(state)
        weights = probs
        # details = [(self.env.action_cls.from_idx(idx), prob) for idx, prob in enumerate(weights)]
        # for action, prob in sorted(details, key=lambda x: x[1], reverse=True):
        #     logging.debug(f"action:{action}, prob:{prob:2.3f}")
        # logging.debug(f"current state value:{value:2.3f}")
        return weights
