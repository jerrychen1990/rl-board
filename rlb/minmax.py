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
import logging
from typing import Tuple, List, Dict

import numpy as np

from rlb.actor_critic import ActorCritic
from rlb.core import Env, Agent, Step
from rlb.utils import weights2probs, sample_by_probs


class Node:
    def __init__(self, obs):
        self.obs = obs
        self.value = None
        self.children: Dict[int, Node] = None

    def is_expanded(self):
        return self.children is not None


class MinMaxAgent(Agent):
    def __init__(self, name, action_num, noise_rate=0.):
        super(MinMaxAgent, self).__init__(name=name, action_num=action_num)
        self.nodes = dict()
        self.noise_rate = noise_rate
        self.is_trained = False

    def get_node(self, obs):
        if obs not in self.nodes:
            node = Node(obs=obs)
            self.nodes[obs] = node
            if len(self.nodes) % 1000 == 0:
                logging.debug(f"{len(self.nodes)} visited")
        return self.nodes[obs]

    def train(self, env: Env):
        def expand_node(node: Node):
            node.children = dict()
            valid_actions = env.get_valid_actions_by_obs(obs=node.obs)

            for action in valid_actions:
                transfer_info = env.transfer(node.obs, action)
                next_node = self.get_node(transfer_info.next_obs)
                if transfer_info.is_done:
                    if transfer_info.extra_info.get("is_win"):
                        next_node.value = -1
                    else:
                        next_node.value = 0
                node.children[action.to_idx()] = transfer_info.reward, next_node

        def get_value(node: Node):
            if node.value is None:
                if not node.is_expanded():
                    expand_node(node)
                assert node.children is not None
                values = []
                for r, n in node.children.values():
                    values.append(-get_value(n))
                value = max(values)
                node.value = value
            return node.value

        obs = env.reset()
        root = self.get_node(obs)
        get_value(root)
        logging.info(f"minmax training done, total {len(self.nodes)} visited")
        self.is_trained = True

    def get_weights(self, obs, mode, **kwargs) -> List[float]:
        node: Node = self.get_node(obs)
        assert node.is_expanded()
        detail = [(a, r - n.value) for a, (r, n) in node.children.items()]
        max_idx, max_w = max(detail, key=lambda x: x[1])

        weights = [0] * self.action_num
        for a, w in detail:
            if w == max_w:
                weights[a] = 1.
        if self.noise_rate:
            logging.info("add noise")
        return weights

    def gen_steps(self, env_cls, step_num=None):
        steps = []
        for idx, (obs, node) in enumerate(self.nodes.items()):
            if not node.is_expanded():
                continue
            weights = self.get_weights(obs=obs, mode="test", mask=None)
            probs = weights2probs(np.array(weights)).tolist()
            action_idx, prob = sample_by_probs(probs)
            action = env_cls.action_cls.from_idx(action_idx)
            transfer_info = env_cls.transfer(obs, action)

            transfer_info.extra_info.update(value=node.value)
            step = Step(agent_name=self.name, obs=obs, action_idx=action_idx, prob=prob,
                        probs=probs, **transfer_info.dict())
            steps.append(step)
            if step_num and idx >= step_num:
                break

        return steps


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
