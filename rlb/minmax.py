#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     minmax.py
   Author :       chenhao
   time：          2022/2/17 16:20
   Description :
-------------------------------------------------
"""
from __future__ import annotations
import logging
import os
import pickle
from typing import Tuple, List, Dict, Optional, Type

import numpy as np
from pydantic import BaseModel
from snippets import pdump, pload

from rlb.actor_critic import ActorCritic
from rlb.core import Env, Agent, Step, ACInfo
from rlb.utils import weights2probs, sample_by_probs

logger = logging.getLogger(__name__)


class Node(BaseModel):
    obs: Tuple
    value: Optional[float]
    children: Optional[Dict[int, Node]]

    def is_expanded(self):
        return self.children is not None

    def __repr__(self):
        return f"{self.obs=},{self.value=},child_num={len(self.children) if self.is_expanded() else None}"


class MinMaxAgent(Agent):
    def __init__(self, name, action_num):
        super(MinMaxAgent, self).__init__(name=name, action_num=action_num)
        self.nodes = dict()
        self.root = None
        self.is_trained = False

    def get_node(self, obs):
        if obs not in self.nodes:
            node = Node(obs=obs)
            self.nodes[obs] = node
            if len(self.nodes) % 1000 == 0:
                logger.debug(f"{len(self.nodes)} visited")
        return self.nodes[obs]

    def _expand_node(self, node: Node, env_cls: Type[Env]):
        node.children = dict()
        valid_actions = env_cls.get_valid_actions_by_obs(obs=node.obs)

        for action in valid_actions:
            transfer_info = env_cls.transfer(node.obs, action)
            next_node = self.get_node(transfer_info.next_obs)
            if transfer_info.is_done:
                win_piece = transfer_info.extra_info.get("win_piece")
                if win_piece:
                    next_node.value = 1 if win_piece == next_node.obs[1] else -1
                else:
                    next_node.value = 0

            node.children[action.to_idx()] = transfer_info.reward, next_node

    def get_value(self, node: Node, env_cls: Type[Env], value_decay: float):
        # logging.info(node)
        if node.value is None:
            if not node.is_expanded():
                self._expand_node(node=node, env_cls=env_cls)
            assert node.children is not None
            children = list(node.children.values())
            values = []
            for r, n in children:
                values.append(-self.get_value(n, env_cls, value_decay) * value_decay)
            max_value = max(values)
            node.value = max_value
        return node.value

    def train(self, env_cls: Type[Env], value_decay=1.):
        if not self.is_trained:
            obs = env_cls().reset()
            cache_path = os.path.join(f"/tmp/rlb_cache/{env_cls.__name__}/{value_decay}.pkl")
            if not os.path.exists(cache_path):
                self.root = self.get_node(obs)
                self.get_value(self.root, env_cls=env_cls, value_decay=value_decay)
                logging.info(f"minmax training done, total {len(self.nodes)} visited")
                pdump(self.nodes, cache_path)
            else:
                self.nodes = pload(cache_path)
                logger.info(f"loaded {len(self.nodes)} minmax nodes from {cache_path} ")
                self.root = self.get_node(obs)
            self.is_trained = True

    def get_weights(self, obs, mode, **kwargs) -> List[float]:
        node: Node = self.get_node(obs)
        if not node.is_expanded():
            # todo handle pass
            return [1] * self.action_num
        weights = [0.] * self.action_num
        min_value = min(n.value for _, n in node.children.values())
        for a, (_, n) in node.children.items():
            if n.value == min_value:
                weights[a] = 1.
        return weights

    def gen_ac_infos(self) -> List[ACInfo]:
        assert self.root
        ac_infos = dict()

        def visit_node(node: Node, is_in_path):
            obs = node.obs
            if obs not in ac_infos:
                if not node.is_expanded():
                    probs = None
                else:
                    weights = self.get_weights(obs=obs, mode="test")
                    probs = weights2probs(np.array(weights)).tolist()
                ac_info = ACInfo(obs=obs, probs=probs, value=node.value, is_in_path=is_in_path)
                ac_infos[obs] = ac_info
            ac_info = ac_infos[obs]
            ac_info.is_in_path = ac_info.is_in_path or is_in_path

            if node.is_expanded():
                for a, (r, n) in node.children.items():
                    iip = ac_info.probs[a] > 0
                    visit_node(n, iip)

        visit_node(self.root, True)
        ac_infos = ac_infos.values()
        return ac_infos


class MinMaxActorCritic(ActorCritic):
    def __init__(self, minmax_agent: MinMaxAgent, action_num, *args, **kwargs):
        super(MinMaxActorCritic, self).__init__(*args, **kwargs)
        self.action_num = action_num
        self.minmax_agent = minmax_agent

    def act_and_criticize(self, obs) -> Tuple[List[float], float]:
        probs = self.act(obs)
        value = self.criticize(obs)
        return probs, value

    def criticize(self, obs) -> float:
        node: Node = self.minmax_agent.get_node(obs)
        value = node.value
        return value

    def act(self, obs) -> List[float]:
        weights = self.minmax_agent.get_weights(obs=obs, mode="test", mask=None)
        assert len(weights) == self.action_num
        return weights2probs(np.array(weights))
