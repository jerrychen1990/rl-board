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
from abc import ABC
from typing import List, Tuple, Type

import numpy as np

from rlb.core import Agent, Env
from rlb.utils import weights2probs


class Critic(ABC):
    def criticize(self, obs) -> float:
        raise NotImplementedError


class Actor(ABC):
    def __init__(self, action_num):
        self.action_num = action_num

    def act(self, obs) -> List[float]:
        raise NotImplementedError


class ActorCritic(Critic, Actor, ABC):
    def act_and_criticize(self, obs) -> Tuple[List[float], float]:
        raise NotImplementedError



class RandomActorCritic(ActorCritic):

    def criticize(self, obs) -> float:
        return random.random()

    def act(self, obs) -> List[float]:
        weights = np.array([random.random() for _ in range(len(self.action_num))])
        probs = weights2probs(weights)
        return probs

    def act_and_criticize(self, obs, valid_actions: List) -> Tuple[List[float], float]:
        v = self.criticize(obs)
        probs = self.act(obs, valid_actions)
        return probs, v


class ActorCriticAgent(Agent):
    def __init__(self, ac_model: ActorCritic, env_cls: Type[Env], *args, **kwargs):
        super(ActorCriticAgent, self).__init__(action_num=ac_model.action_num, *args, **kwargs)
        self.ac_model = ac_model
        self.env_cls = env_cls
        assert self.ac_model.action_num == self.env_cls.action_num

    def get_weights(self, obs, mode, **kwargs) -> List[float]:
        logging.debug(obs)
        probs, value = self.ac_model.act_and_criticize(obs)
        weights = probs
        details = [(self.env_cls.action_cls.from_idx(idx), prob) for idx, prob in enumerate(weights)]
        for action, prob in sorted(details, key=lambda x: x[1], reverse=True):
            logging.debug(f"action:{action}, prob:{prob:2.3f}")
        logging.debug(f"current state value:{value:2.3f}")
        return weights
