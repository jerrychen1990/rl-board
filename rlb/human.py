#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     human.py
   Author :       chenhao
   time：          2022/2/17 16:07
   Description :
-------------------------------------------------
"""
import logging
from typing import List, Type

from rlb.core import Agent, Action, State


class HumanAgent(Agent):
    def __init__(self, action_cls: Type[Action], *args, **kwargs):
        super(HumanAgent, self).__init__(*args, **kwargs)
        self.action_cls = action_cls

    def get_weights(self, state: State, **kwargs) -> List[float]:
        while True:
            try:
                logging.info("input action:")
                cmd = input().strip()
                action_idx = self._cmd2action_idx(cmd)
                probs = [0] * self.action_num
                probs[action_idx] = 1.
                return probs
            except Exception as e:
                logging.exception(e)

    def _cmd2action_idx(self, cmd: str) -> int:
        return self.action_cls.from_cmd(cmd)
