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
from abc import abstractmethod
from typing import List

from rlb.core import Agent
from rlb.tictactoe import TicTacToe
from rlb.gravity_gobang import GravityGoBang, GravityGoBangAction


class AbsHumanAgent(Agent):
    def get_weights(self, obs, mode, **kwargs) -> List[float]:
        while True:
            try:
                logging.info("input action:")
                cmd = input().strip()
                action_idx = self._cmd2action(cmd)
                probs = [0] * self.action_num
                probs[action_idx] = 1.
                return probs
            except Exception as e:
                logging.exception(e)

    @abstractmethod
    def _cmd2action(self, cmd):
        pass


class TicTacToeHumanAgent(AbsHumanAgent):
    def _cmd2action(self, cmd):
        r, c = cmd.split(",")
        r, c = int(r.strip()), int(c.strip())
        action = TicTacToe.TicTacToeAction(r=r, c=c)
        return action.to_idx()


class GravityGoBangHumanAgent(AbsHumanAgent):
    def _cmd2action(self, cmd):
        c = int(cmd.strip())
        action = GravityGoBangAction(c=c)
        return action.to_idx()
