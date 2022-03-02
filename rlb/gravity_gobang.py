#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     gravity_gobang.py
   Author :       chenhao
   time：          2022/2/25 17:25
   Description :
-------------------------------------------------
"""
import logging
import types
from abc import ABC
from typing import List, Tuple

from rlb.core import Board, Piece, Action, BoardEnv, TransferInfo

logger = logging.getLogger(__name__)


class GravityGoBangAction(Action):
    c: int

    def to_idx(self) -> int:
        return self.c

    @classmethod
    def from_idx(cls, idx):
        return cls(c=idx)

    def __str__(self):
        return f"[{self.c}]"


class GravityGoBang(BoardEnv, ABC):
    action_cls = GravityGoBangAction
    target_size: int = NotImplemented

    @staticmethod
    def _get_first_blank(line):
        for idx, p in enumerate(line):
            if p == Piece.BLANK:
                return idx
        return len(line)

    @classmethod
    def _get_valid_actions_by_board_piece(cls, board: Board, piece: Piece) -> List[GravityGoBangAction]:
        return [cls.action_cls(c=c) for c, col in enumerate(board.cols)
                if cls._get_first_blank(col[::-1]) != cls.board_size]

    @classmethod
    def _get_next_piece(cls, next_piece):
        if next_piece == Piece.X:
            next_piece = Piece.O
        elif next_piece == Piece.O:
            next_piece = Piece.X
        else:
            logger.error(f"invalid piece:{next_piece}")
        return next_piece

    @classmethod
    def _get_r_by_c(cls, board: Board, c: int) -> int:
        col = board.get_col(c)
        return cls.board_size - GravityGoBang._get_first_blank(col[::-1]) - 1

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
        def is_same_line(line, n):
            pre = None
            idx = 0
            acc = 0
            while idx < len(line):
                if line[idx] == Piece.BLANK:
                    acc = 0
                else:
                    if pre is None or line[idx] == pre:
                        acc += 1
                        if acc >= n:
                            return True
                    else:
                        acc = 1
                pre = line[idx]
                idx += 1
            return acc >= n

        for l in board.rows + board.cols:
            if is_same_line(l, cls.target_size):
                return True
        for l in board.slashes:
            if is_same_line(l, cls.target_size):
                return True
        return False

    @classmethod
    def _on_invalid_action(cls, board: Board, piece: Piece,
                           action: GravityGoBangAction) -> Tuple[TransferInfo, Piece]:
        logger.info(f"invalid action:{action}")
        next_obs = cls._get_obs(board, piece)
        next_piece = piece
        transfer_info = TransferInfo(next_obs=next_obs, reward=-10, is_done=False, extra_info=dict(valide=False))

        return transfer_info, next_piece

    @classmethod
    def _transfer_on_board_piece(cls, board: Board, piece: Piece,
                                 action: GravityGoBangAction) -> Tuple[TransferInfo, Piece]:
        c = action.c
        r = cls._get_r_by_c(board, c)
        assert 0 <= r < cls.board_size and 0 <= c < cls.board_size
        board.set_piece(r, action.c, piece)
        next_piece = cls._get_next_piece(piece)
        reward, is_done, info = cls.judge_board(board)
        next_obs = cls._get_obs(board, next_piece)
        transfer_info = TransferInfo(next_obs=next_obs, reward=reward, is_done=is_done, extra_info=info)
        return transfer_info, next_piece


def build_gravity_gobang_cls(name, board_size: int, target_size: int) -> type:
    attrs = dict(board_size=board_size, target_size=target_size, action_num=board_size)
    cls = types.new_class(name=name, bases=(GravityGoBang,), kwds={}, exec_body=lambda x: x.update(**attrs))
    cls.__module__ = __name__
    return cls


GravityGoBang3 = build_gravity_gobang_cls(name="GravityGoBang3", board_size=4, target_size=3)
# print(__name__)
# print(GravityGoBang3.__dict__)
# print(GravityGoBang3.__module__)

