#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     gobang.py
   Author :       chenhao
   time：          2022/2/25 17:25
   Description :
-------------------------------------------------
"""
import copy
import logging
import types
from abc import abstractmethod
from typing import List, Tuple

from rlb.board_core import Board, Piece, Action, BoardEnv, TransferInfo, BoardState, BoardAction
from rlb.core import State

logger = logging.getLogger(__name__)


class GravityGoBangAction(Action):
    c: int

    def to_idx(self) -> int:
        return self.c

    @classmethod
    def from_idx(cls, idx):
        return cls(c=idx)

    @classmethod
    def from_cmd(cls, cmd: str) -> int:
        c = int(cmd.strip())
        action = GravityGoBangAction(c=c)
        return action.to_idx()

    @classmethod
    def get_pass_action(cls) -> Action:
        return None

    def __str__(self):
        return f"[{self.c}]"


class BaseGoBang(BoardEnv):
    pieces = [Piece.X, Piece.O]
    target_size: int = NotImplemented
    draw2loss: bool = NotImplemented

    @classmethod
    def judge_board(cls, board: Board, piece: Piece):
        is_win = cls.is_win(board)
        is_full = board.is_full()
        is_done = is_win or is_full

        if is_win:
            win_piece = piece
        elif is_full and cls.draw2loss:
            win_piece = cls.pieces[-1]
        else:
            win_piece = None

        if is_win:
            reward = 10
        else:
            reward = -0.1
        return reward, is_done, dict(win_piece=win_piece)

    @classmethod
    def get_win_piece(cls, board):
        def is_same_line(line, n):
            pre = None
            idx = 0
            acc = 0
            if len(line) < n:
                return False
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
                return l[0]
        for l in board.slashes:
            if is_same_line(l, cls.target_size):
                return l[0]
        return None

    @classmethod
    @abstractmethod
    def _get_rc_by_action(cls, action: Action, board: Board) -> Tuple[int, int]:
        raise NotImplementedError

    @classmethod
    def _on_valid_action(cls, state: BoardState, action: Action) -> TransferInfo:
        r, c = cls._get_rc_by_action(action=action, board=state.board)
        assert 0 <= r < cls.board_size and 0 <= c < cls.board_size
        next_state = copy.deepcopy(state)
        next_state.board.set_piece(r, c, state.piece)
        next_state.piece = cls._get_next_piece(state.piece)
        next_state.pass_num = 0

        extra_info = dict()
        win_piece = cls.get_win_piece(next_state.board)
        if win_piece:
            is_done = True
            extra_info.update(win_piece=win_piece)
        else:
            if next_state.board.is_full():
                is_done = True
                extra_info.update(win_piece=next_state.piece if cls.draw2loss else None)
            else:
                is_done = False
        transfer_info = TransferInfo(next_state=next_state, reward=0., is_done=is_done, extra_info=extra_info)
        return transfer_info


class GoBang(BaseGoBang):
    @classmethod
    def _on_invalid_action(cls, state: BoardState, action: Action) -> TransferInfo:
        logger.debug(f"invalid action:{action}, regard as pass")
        return cls._on_pass_action(state)

    @classmethod
    def get_valid_actions_by_state(cls, state: State) -> List[Action]:
        return [cls.action_cls(r=r, c=c) for r in range(cls.board_size) for c in range(cls.board_size)
                if state.board.get_piece(r, c) == Piece.BLANK] + [cls.action_cls.get_pass_action()]

    @classmethod
    def _on_pass_action(cls, state: BoardState) -> TransferInfo:
        next_state = copy.deepcopy(state)
        next_state.piece = cls._get_next_piece(state.piece)
        next_state.pass_num += 1
        is_done = next_state.pass_num >= len(cls.pieces)
        transfer_info = TransferInfo(next_state=next_state, reward=0., is_done=is_done, extra_info=dict(is_pass=True))
        return transfer_info

    @classmethod
    def _get_rc_by_action(cls, action: BoardAction, board: Board) -> Tuple[int, int]:
        r, c = action.r, action.c
        return r, c


class GravityGoBang(BaseGoBang):

    @classmethod
    def _on_invalid_action(cls, state: BoardState, action: GravityGoBangAction) -> TransferInfo:
        logger.debug(f"invalid action :{action}, will retry")
        next_state = copy.deepcopy(state)
        transfer_info = TransferInfo(next_state=next_state, reward=0., is_done=False, extra_info=dict(valid=False))
        return transfer_info

    @classmethod
    def get_valid_actions_by_state(cls, state: BoardState) -> List[GravityGoBangAction]:
        return [cls.action_cls(c=c) for c, col in enumerate(state.board.cols)
                if cls._get_first_blank(col[::-1]) != cls.board_size]

    @classmethod
    def _on_pass_action(cls, state: BoardState) -> TransferInfo:
        raise ValueError("GravityGoBang don't allow pass!")

    @staticmethod
    def _get_first_blank(line):
        for idx, p in enumerate(line):
            if p == Piece.BLANK:
                return idx
        return len(line)

    @classmethod
    def _get_rc_by_action(cls, action: GravityGoBangAction, board: Board) -> Tuple[int, int]:
        c = action.c
        col = board.get_col(c)
        r = cls.board_size - GravityGoBang._get_first_blank(col[::-1]) - 1
        return r, c


def build_gobang_cls(name, board_size: int, target_size: int, draw2loss: bool, gravity: bool) -> type:
    base_class = GravityGoBang if gravity else GoBang
    action_num = board_size if gravity else board_size ** 2 + 1
    if gravity:
        action_cls = GravityGoBangAction
    else:
        action_cls = types.new_class(name=f"{name}Action", bases=(BoardAction,),
                                     kwds={}, exec_body=lambda x: x.update(_board_size=board_size))

    attrs = dict(board_size=board_size, target_size=target_size, action_num=action_num, draw2loss=draw2loss,
                 action_cls=action_cls)
    cls = types.new_class(name=name, bases=(base_class,), kwds={}, exec_body=lambda x: x.update(**attrs))
    cls.__module__ = __name__
    return cls


GravityGoBang34 = build_gobang_cls(name="GravityGoBang34", board_size=4, target_size=3, draw2loss=True, gravity=True)
GravityGoBang33 = build_gobang_cls(name="GravityGoBang33", board_size=3, target_size=3, draw2loss=True, gravity=True)
GravityGoBang47 = build_gobang_cls(name="GravityGoBang47", board_size=7, target_size=4, draw2loss=False, gravity=True)

TicTacToe = build_gobang_cls(name="TicTacToe", board_size=3, target_size=3, draw2loss=False, gravity=False)

__all__env_cls__ = [GravityGoBang33, GravityGoBang34, GravityGoBang47, TicTacToe]
