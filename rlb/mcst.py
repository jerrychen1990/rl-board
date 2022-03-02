#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     mcst.py
   Author :       chenhao
   time：          2022/2/15 11:42
   Description :
-------------------------------------------------
"""
import logging
import math
from typing import List, Tuple, Dict, Type, Callable

import numpy as np
from pydantic import BaseModel, Field
from torch.nn import Module

from rlb.actor_critic import ModuleActorCritic
from rlb.core import Agent, Env, TransferInfo, Action
from rlb.utils import sample_by_weights

logger = logging.getLogger(__name__)


class Info(BaseModel):
    n: int = Field(description="visit num", default=0)
    p: float = Field(description="prior probability", le=1., ge=0.)
    w: float = Field(description="value", default=0)

    @property
    def q(self):
        return self.w / self.n if self.n else 0.

    def __str__(self):
        return f"Info(n={self.n}, p={self.p:2.3f}, w={self.w:2.3f}, q={self.q:2.3f})"


def putc(q, n, n_total, p, c, noise_rate=0, noise=0):
    if noise_rate:
        p = (1 - noise_rate) * p + noise_rate * noise

    u = c * p * math.sqrt(n_total+1) / (1 + n)
    return q + u


def visit_tau(n, tau):
    try:
        return min(math.pow(n, tau), 1e9)
    except Exception as e:
        raise e


class Node:
    def __init__(self, obs):
        self.obs = obs
        self.children: Dict[Action, Tuple[Info, Node]] = None
        self.value = 0.
        self.is_done = False

    def is_expanded(self):
        return self.children is not None

    def is_leaf(self):
        if self.is_done:
            return True
        assert self.is_expanded()
        return len(self.children) == 0

    def get_choose_child_detail(self, mode, noises, noise_rate=0, tau=1., **kwargs):
        assert self.is_expanded()
        details = []
        n_total = sum([i.n for i, node in self.children.values()])

        for action, (info, node) in self.children.items():
            if mode == "putc":
                noise = noises[action.to_idx()]
                weight = putc(q=info.q, n=info.n, p=info.p, n_total=n_total,
                              noise=noise, noise_rate=noise_rate, **kwargs)
            elif mode == "visit_tau":
                weight = visit_tau(n=info.n, tau=tau)
            else:
                raise ValueError(f"invalid mode:{mode}")
            details.append((action, info, node, weight))

        if logger.level == logging.DEBUG:
            details.sort(key=lambda x: x[-1], reverse=True)
            logger.debug(f"details({mode}):")
            for action, info, node, weight in details:
                logger.debug(f"action:{action}, info:{info}, noise:{noises[action.to_idx()]:2.3f},"
                             f" tau:{tau:1.1f}, weight:{weight:2.3f}, next_node:{node}")
        return details

    def __str__(self):
        return f"{self.obs}[{self.value:2.3f}]"

    def __repr__(self):
        return str(self)


class MCST:
    def __init__(self, transfer_func: Callable, valid_action_func: Callable, action_num, c=2, noise_kwargs=dict()):
        self.node_dict = dict()
        self.c = c
        self.action_num = action_num
        self.transfer_func = transfer_func
        self.valid_action_func = valid_action_func
        self.noise_kwargs = noise_kwargs
        self.noise_rate = noise_kwargs.get("noise_rate", 0)
        self.dirichlet_alpha = noise_kwargs.get("dirichlet_alpha", 0)

    def get_node(self, obs) -> Node:
        if obs not in self.node_dict:
            node = Node(obs=obs)
            self.node_dict[obs] = node
        return self.node_dict[obs]

    def _select(self, node: Node):
        trace = []
        while node.is_expanded() and not node.is_leaf():
            if self.noise_rate:
                noises = np.random.dirichlet([self.dirichlet_alpha] * self.action_num)
            else:
                noises = [0]*self.action_num

            details = node.get_choose_child_detail(mode="putc", noises=noises, noise_rate=self.noise_rate, c=self.c)
            weights = np.array([e[-1] for e in details])

            idx, prob = sample_by_weights(weights, deterministic=True)
            action, info, next_node, weight = details[idx]
            trace.append((node, action, prob, next_node))
            node = next_node
        return node, trace

    def _expand_eval(self, node: Node, ac_model):
        action_probs, value = ac_model.act_and_criticize(obs=node.obs)
        if not node.is_done:
            node.children = dict()
            valid_actions = self.valid_action_func(obs=node.obs)
            if valid_actions:
                action_probs, value = ac_model.act_and_criticize(obs=node.obs)
                for action in valid_actions:
                    transfer_info: TransferInfo = self.transfer_func(obs=node.obs, action=action)
                    next_obs = transfer_info.next_obs
                    next_node = self.get_node(next_obs)
                    next_node.is_done = transfer_info.is_done
                    prob = action_probs[action.to_idx()]
                    node.children[action] = Info(p=prob), next_node
        node.value = value
        logger.debug(f"set value:{value:2.3f} to node:{node}")

        return value

    @classmethod
    def _backward(cls, trace, value):
        w = -value
        for node, action, prob, next_node in trace[::-1]:
            info = node.children[action][0]
            info.n += 1
            info.w += w
            w = - w

    def simulate(self, node: Node, ac_model):
        logger.debug("selecting")
        leaf_node, trace = self._select(node)
        logger.debug("expanding")
        value = self._expand_eval(node=leaf_node, ac_model=ac_model)
        logger.debug("backwarding")
        self._backward(trace, value)

    def clear(self):
        # logger.info("clear mcst")
        self.node_dict = dict()


class TauSchedule:
    def __init__(self, tau: float, schedule: list = []):
        self._tau = tau
        self._schedule = schedule

    def get_tau(self, mode, step_idx):
        t = 1 / self._tau
        if mode == "test":
            return t * 3
        else:
            alpha = 1
            for k, v in self._schedule:
                if step_idx < v:
                    break
                else:
                    alpha = alpha
            return t * alpha


class MCSTAgent(Agent):
    def __init__(self, env_cls: Type[Env], ac_model: ModuleActorCritic, simulate_num,
                 tau_kwargs=dict(tau=1.), c=2, noise_kwargs={}, **kwargs):
        super(MCSTAgent, self).__init__(action_num=ac_model.action_num, **kwargs)
        self.env_cls = env_cls
        self.ac_model = ac_model
        self.mcst = MCST(transfer_func=env_cls.transfer, valid_action_func=env_cls.get_valid_actions_by_obs,
                         c=c, noise_kwargs=noise_kwargs, action_num=self.action_num)
        self.tau_schedule = TauSchedule(**tau_kwargs)
        self.simulate_num = simulate_num

    def get_weights(self, obs, mode, step_idx, **kwargs) -> List[float]:
        node = self.mcst.get_node(obs)
        logger.debug(f"simulating for {self.simulate_num} times...")
        for idx in range(self.simulate_num):
            logger.debug(f"simulation:{idx + 1}")
            self.mcst.simulate(node, ac_model=self.ac_model)

        tau = self.tau_schedule.get_tau(mode, step_idx)

        details = node.get_choose_child_detail(mode="visit_tau", tau=tau, noises=[0]*self.action_num)
        weights = [0.] * self.action_num
        for action, info, node, weight in details:
            action_idx = action.to_idx()
            weights[action_idx] = weight
        return weights

    def update_model(self, ac_model: Module):
        # logger.info("replace ac_model")
        self.ac_model.load_state_dict(ac_model.state_dict())
        # for p in ac_model.parameters():
        #     logger.info(p[0][:10])
        #     break
        self.clear_mcst()

    def clear_mcst(self):
        self.mcst.clear()

    def on_episode_end(self, episode_idx):
        self.clear_mcst()
