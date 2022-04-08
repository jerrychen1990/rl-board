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
logger = logging.getLogger(__name__)


class HumanAgent(Agent):
    def __init__(self, board_size: int, *args, **kwargs):
        super(HumanAgent, self).__init__(*args, **kwargs)
        self.board_size = board_size

    def get_weights(self, state: State, mode: str, **kwargs) -> List[float]:
        while True:
            try:
                logging.info("input action:")
                cmd = input().strip()
                if cmd.upper() == "P":
                    action_idx = self.board_size ** 2
                    if action_idx >= self.action_num:
                        raise Exception("env not allow pass!")

                else:
                    r, c = cmd.split(",")
                    r, c = int(r), int(c)
                    action_idx = r*self.board_size+c
                probs = [0.] * self.action_num
                probs[action_idx] = 1.
                return probs
            except Exception as e:
                logger.error(e)
