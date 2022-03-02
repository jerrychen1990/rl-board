#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     tictactoe.py
   Author :       chenhao
   time：          2022/2/15 10:39
   Description :
-------------------------------------------------
"""
import logging
from typing import List, Tuple

from rlb.core import Env, Piece, Board, Action

logger = logging.getLogger(__name__)


class TicTacToe(Env):
    board_size = 3
    nA = board_size ** 2

    class TicTacToeAction(Action):
        r: int
        c: int

        def to_idx(self) -> int:
            return self.r * TicTacToe.board_size + self.c

        @classmethod
        def from_idx(cls, idx):
            r, c = divmod(idx, TicTacToe.board_size)
            return cls(r=r, c=c)

        def __str__(self):
            return f"[{self.r}, {self.c}]"

    action_cls = TicTacToeAction

    def __init__(self):
        self._reset()

    def _reset(self):
        self._board = Board(row_num=self.board_size, col_num=self.board_size)
        self._next_piece = Piece.X

    def reset(self):
        self._reset()
        return self.get_obs()

    @property
    def action_num(self):
        return self.board_size ** 2

    @classmethod
    def _get_next_piece(cls, next_piece):
        if next_piece == Piece.X:
            next_piece = Piece.O
        elif next_piece == Piece.O:
            next_piece = Piece.X
        else:
            logger.error(f"invalid piece:{next_piece}")
        return next_piece

    def _change_piece(self):
        self._next_piece = self._get_next_piece(self._next_piece)

    @classmethod
    def _get_valid_actions_by_board(cls, board) -> List:
        reward, is_done, info = cls.judge_board(board)
        if is_done:
            return []
        return [cls.TicTacToeAction(r=r, c=c) for r in range(cls.board_size)
                for c in range(cls.board_size) if board.get_piece(r, c) == Piece.BLANK]

    @classmethod
    def get_valid_actions_by_obs(cls, obs) -> List:
        board, _ = cls._obs2board_piece(obs)
        return cls._get_valid_actions_by_board(board)

    def get_valid_actions(self) -> List:
        return self._get_valid_actions_by_board(self._board)

    @classmethod
    def transfer(cls, obs, action):
        row, col = action.r, action.c
        board, next_piece = cls._obs2board_piece(obs)
        board.set_piece(row, col, next_piece)
        next_piece = cls._get_next_piece(next_piece)
        reward, is_done, info = cls.judge_board(board)
        next_obs = cls._get_obs(board, next_piece)
        return next_obs, reward, is_done, info

    def step(self, action):
        row, col = action.r, action.c
        self._board.set_piece(row, col, self._next_piece)
        self._change_piece()
        reward, is_done, info = self.judge_board(self._board)
        next_obs = self.get_obs()
        return next_obs, reward, is_done, info

    @classmethod
    def judge_board(cls, board: Board):
        is_win = cls.is_win(board)
        is_full = board.is_full()

        is_done = is_win or is_full
        if is_win:
            reward = 10
        else:
            reward = -0.1
        return reward, is_done, dict(is_win=is_win, is_full=is_full)

    @classmethod
    def is_win(cls, board):
        def is_same_line(line):
            pre = None
            for e in line:
                if e == Piece.BLANK or (pre and pre != e):
                    return False
                pre = e
            return True

        for l in board.rows + board.cols:
            if is_same_line(l):
                return True

        for l in board.diagonals:
            if is_same_line(l):
                return True
        return False

    def get_obs(self):
        return self._get_obs(self._board, self._next_piece)

    @classmethod
    def _get_obs(cls, board, next_piece):
        obs = []
        for row in board.rows:
            obs.append(tuple(row))
        return tuple(obs), next_piece

    @classmethod
    def _obs2board_piece(cls, obs) -> Tuple[Board, Piece]:
        b, next_piece = obs
        b = [list(e) for e in b]
        board = Board(row_num=cls.board_size, col_num=cls.board_size, board=b)
        return board, next_piece

    def render(self, mode="human"):
        state_info = f"\n{self._board}|{self._next_piece}"
        logger.info(state_info)

    def close(self):
        pass

    def seed(self, seed):
        pass
