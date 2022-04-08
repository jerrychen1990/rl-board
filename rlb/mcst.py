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
import copy
import logging
import math
from typing import List, Tuple, Dict, Type, Callable

import numpy as np
from pydantic import BaseModel, Field
from snippets import discard_kwarg
from torch.nn import Module

from rlb.model import ModuleActorCritic
from rlb.core import Agent, BoardEnv, TransferInfo, Action, State
from rlb.utils import sample_by_weights, format_dict

logger = logging.getLogger(__name__)


class Info(BaseModel):
    n: int = Field(description="visit num", default=0)
    p: float = Field(description="prior probability", le=1., ge=0.)
    w: float = Field(description="value", default=0)

    @property
    def q(self):
        return self.w / self.n if self.n else 0.

    def __str__(self):
        return f"Info(n={self.n:2d}, p={self.p:2.3f}, w={self.w: 2.3f}, q={self.q:+2.3f})"


def putc(q, n, n_total, p, c):
    u = c * p * math.sqrt(n_total + 1) / (1 + n)
    return q + u


def visit_tau(n, tau):
    try:
        return min(math.pow(n, tau), 1e9)
    except Exception as e:
        raise e


class Node:
    def __init__(self, state: State):
        self.state = state
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

    def get_choose_child_detail(self, mode, noise_kwargs: dict = {}, ignore_prob=False, tau=1., c=1):
        assert self.is_expanded()
        details = []
        n_total = sum([i.n for i, node in self.children.values()])
        action_num = len(self.children)
        noise_rate = noise_kwargs.get("noise_rate", 0.)
        if noise_rate:
            noises = np.random.dirichlet([noise_kwargs["dirichlet_alpha"]] * action_num)

        for idx, (action, (info, node)) in enumerate(self.children.items()):
            extra_info = dict()
            if mode == "putc":
                if ignore_prob:
                    p = 1 / action_num
                else:
                    p = info.p
                    if noise_rate:
                        p = (1 - noise_rate) * p + noise_rate * noises[idx]
                        extra_info.update(noise=noises[idx])
                extra_info.update(p=p, c=c)
                weight = putc(q=info.q, n=info.n, p=p, n_total=n_total, c=c)
            elif mode == "visit_tau":
                weight = visit_tau(n=info.n, tau=tau)
                extra_info.update(tau=tau)
            else:
                raise ValueError(f"invalid mode:{mode}")
            details.append((action, info, node, extra_info, weight))

        if logger.level == logging.DEBUG:
            details.sort(key=lambda x: x[-1], reverse=True)
            logger.debug(f"choose child for {self} with {mode} mode")
            for action, info, node, extra_info, weight in details:
                logger.debug(f"action:{action}, info:{info},  extra_info:{format_dict(extra_info)}, "
                             f"weight:{weight:8.3f}, next_node:{node}")
        return details

    def __repr__(self):
        s_str = str(self.state)
        return f"{s_str}[{self.value:2.3f}]"



class MCST:
    def __init__(self, transfer_func: Callable, valid_action_func: Callable, action_num, c=2, noise_kwargs=dict()):
        self.node_dict = dict()
        self.c = c
        self.action_num = action_num
        self.transfer_func = transfer_func
        self.valid_action_func = valid_action_func
        self.noise_kwargs = noise_kwargs

    def get_node(self, state) -> Node:
        if state not in self.node_dict:
            node = Node(state=state)
            self.node_dict[state] = node
        return self.node_dict[state]

    @discard_kwarg
    def _select(self, node: Node, c, ignore_prob=False, noise_kwargs={}):
        trace = []
        while node.is_expanded() and not node.is_leaf():
            details = node.get_choose_child_detail(mode="putc", noise_kwargs=noise_kwargs, c=c, ignore_prob=ignore_prob)
            weights = np.array([e[-1] for e in details])
            idx, prob = sample_by_weights(weights, deterministic=True)
            action, info, next_node, weight, extra_info = details[idx]
            trace.append((node, action, prob, next_node))
            node = next_node
        return node, trace

    def _expand_eval(self, node: Node, ac_model):
        action_probs, value = ac_model.act_and_criticize(state=node.state)
        if not node.is_done:
            node.children = dict()
            valid_actions = self.valid_action_func(state=node.state)
            if valid_actions:
                action_probs, value = ac_model.act_and_criticize(state=node.state)
                for action in valid_actions:
                    transfer_info: TransferInfo = self.transfer_func(state=node.state, action=action)
                    next_node = self.get_node(transfer_info.next_state)
                    next_node.is_done = transfer_info.is_done
                    prob = action_probs[action.idx]
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

    def simulate(self, node: Node, ac_model, **kwargs):
        logger.debug("selecting")
        leaf_node, trace = self._select(node, **kwargs)
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
                if step_idx < k:
                    break
                else:
                    alpha = v
            return t * alpha


class MCSTAgent(Agent):
    def __init__(self, env: BoardEnv, ac_model: ModuleActorCritic, simulate_kwargs: dict,
                 tau_kwargs=dict(tau=1.), **kwargs):
        super(MCSTAgent, self).__init__(action_num=ac_model.action_num, **kwargs)
        self.env = env
        self.ac_model = ac_model
        self.mcst = MCST(transfer_func=env.transfer, valid_action_func=env.get_valid_actions,
                         action_num=self.action_num)
        self.simulate_kwargs = copy.copy(simulate_kwargs)
        self.simulate_num = self.simulate_kwargs.pop("simulate_num")
        self.tau_schedule = TauSchedule(**tau_kwargs)

    def get_weights(self, state: State, mode: str, step_idx, **kwargs) -> List[float]:
        node = self.mcst.get_node(state)
        logger.debug(f"simulating for {self.simulate_num} times...")
        simulate_kwargs = copy.copy(self.simulate_kwargs)
        if mode == "test":
            simulate_kwargs["noise_rate"] = 0.

        for idx in range(self.simulate_num):
            logger.debug(f"simulation:{idx + 1}")
            self.mcst.simulate(node, ac_model=self.ac_model, **simulate_kwargs)

        tau = self.tau_schedule.get_tau(mode, step_idx)
        logger.debug(f"{tau=:2.3f}")

        details = node.get_choose_child_detail(mode="visit_tau", tau=tau)
        weights = [0.] * self.action_num
        for action, info, node, extra_info, weight in details:
            action_idx = action.idx
            weights[action_idx] = weight
        # logger.info(weights)
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
