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
from typing import Tuple, List, Dict, Optional, Type, Set

import numpy as np
from pydantic import BaseModel
from snippets import pdump, pload

from rlb.actor_critic import ActorCritic
from rlb.core import BoardEnv, Agent, State, ACInfo
from rlb.utils import weights2probs

logger = logging.getLogger(__name__)


class Node(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    state: State
    value: Optional[float]
    children: Optional[Dict[int, Node]]
    max_actions: Set[int] = None
    is_in_path = False

    @property
    def piece(self):
        return self.state.piece

    def is_expanded(self):
        return self.children is not None

    def __repr__(self):
        return f"{self.state=},{self.value=},child_num={len(self.children) if self.is_expanded() else None}"


class MinMaxAgent(Agent):
    def __init__(self, name, action_num):
        super(MinMaxAgent, self).__init__(name=name, action_num=action_num)
        self.nodes = dict()
        self.root = None
        self.is_trained = False

    def creat_node(self, state):
        node = Node(state=state)
        self.nodes[state] = node
        if len(self.nodes) % 5000 == 0:
            logger.info(f"{len(self.nodes)} created")
        return node

    def get_node(self, state):
        if state in self.nodes:
            return self.nodes[state]
        return self.creat_node(state)

    def _expand_node(self, node: Node, env: BoardEnv):
        node.children = dict()
        valid_actions = env.get_valid_actions(state=node.state)

        for action in valid_actions:
            transfer_info = env.transfer(node.state, action)
            next_state = transfer_info.next_state
            next_node = self.get_node(next_state)
            if transfer_info.is_done:
                win_piece = transfer_info.win_piece
                if win_piece:
                    next_node.value = 1 if win_piece == next_node.piece else -1
                else:
                    next_node.value = 0
            node.children[action.idx] = next_node

    def get_value(self, node: Node, env: BoardEnv, value_decay: float, stack: List) -> float:
        # logging.info(node)
        if node.value is None:
            stack.append(node.state)

            if not node.is_expanded():
                self._expand_node(node=node, env=env)

            max_value = -1.
            max_actions = set()

            for action_idx, n in node.children.items():
                if n.state in stack:
                    continue
                next_value = self.get_value(n, env, value_decay, stack)
                value = next_value if node.piece == n.piece else -next_value
                value *= value_decay
                if value >= max_value:
                    if value == max_value:
                        max_actions.add(action_idx)
                    else:
                        max_actions = {action_idx}
                    max_value = value
            node.value = max_value
            node.max_actions = max_actions
            for action in max_actions:
                node.children[action].is_in_path = True
            stack.pop()
        return node.value

    def train(self, env: BoardEnv, value_decay=1., overwrite_cache=False):
        if not self.is_trained:
            state = env.reset()
            self.nodes = dict()
            cache_path = os.path.join(f"/tmp/rlb_cache/{env.name}/{value_decay}.pkl")
            if not os.path.exists(cache_path) or overwrite_cache:
                self.root = self.get_node(state)
                self.get_value(self.root, env=env, value_decay=value_decay, stack=[])
                logger.info(f"minmax training done, dumping to {cache_path}")
                pdump(self.nodes, cache_path)
            else:
                logging.info(f"loading trained nodes from {cache_path}")
                self.nodes = pload(cache_path)
                self.root = self.get_node(state)
            logger.info(f"{len(self.nodes)} nodes trained")
            self.is_trained = True

    def get_weights(self, state, mode, **kwargs) -> List[float]:
        assert self.is_trained
        assert state in self.nodes

        node: Node = self.nodes[state]
        assert node.is_expanded()
        weights = [0.] * self.action_num
        for action in node.max_actions:
            weights[action] = 1.
        return weights

    def gen_ac_infos(self) -> List[ACInfo]:
        assert self.is_trained
        ac_infos = []
        for state, node in self.nodes.items():
            if not node.is_expanded():
                probs = None
            else:
                weights = self.get_weights(state, mode="test")
                sum_weights = sum(weights)
                assert sum_weights > 0
                probs = [e / sum_weights for e in weights]
            ac_info = ACInfo(state=state, value=node.value, probs=probs, is_in_path=node.is_in_path)
            ac_infos.append(ac_info)
        return ac_infos


class MinMaxActorCritic(ActorCritic):
    def __init__(self, minmax_agent: MinMaxAgent, action_num, *args, **kwargs):
        super(MinMaxActorCritic, self).__init__(*args, **kwargs)
        self.action_num = action_num
        self.minmax_agent = minmax_agent

    def act_and_criticize(self, state) -> Tuple[List[float], float]:
        probs = self.act(state)
        value = self.criticize(state)
        return probs, value

    def criticize(self, state) -> float:
        node: Node = self.minmax_agent.get_node(state)
        value = node.value
        return value

    def act(self, state) -> List[float]:
        weights = self.minmax_agent.get_weights(state=state, mode="test", mask=None)
        assert len(weights) == self.action_num
        return weights2probs(np.array(weights))
