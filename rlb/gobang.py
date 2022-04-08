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
from itertools import chain
from typing import List, Tuple
from rlb.core import *

logger = logging.getLogger(__name__)


class GoBang(BoardEnv):
    pieces = [Piece.X, Piece.O]
    allow_pass = False

    def __init__(self, target_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_size = target_size

    def reset(self) -> State:
        board = Board.get_empty_board(self.board_size)
        state = State.from_board_piece(board=board, piece=self.pieces[0])
        return state

    def judge_state(self, state: State) -> Edge:
        board = state.get_board()
        win_piece = self.get_win_piece(board)
        if win_piece:
            is_done = True
        else:
            is_done = board.is_full()
        return Edge(reward=0., is_done=is_done, win_piece=win_piece)

    def get_valid_actions(self, state: State) -> List[Action]:
        actions = []
        board, piece = state.get_board(), state.piece
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board.get_piece(r, c) == Piece.B:
                    actions.append(Action(idx=r * self.board_size + c, r=r, c=c, p=piece))
        return actions

    def get_win_piece(self, board: Board) -> Piece:
        for l in board.gen_all_lines():
            win_piece = get_continue_piece(l, self.target_size)
            if win_piece:
                return win_piece
        return None

    def on_valid_action(self, state: State, action: Action) -> TransferInfo:
        next_board, piece = state.to_new_board(), state.piece
        next_board.set_piece(action.r, action.c, piece)
        next_piece = self.get_next_piece(piece)
        next_state = State.from_board_piece(next_board, next_piece)
        edge = self.judge_state(next_state)
        transfer_info = TransferInfo(next_state=next_state, **edge.dict(), extra_info=dict())
        return transfer_info


def get_continue_piece(line: List[Piece], tgt: int) -> Piece:
    pre = None
    acc = 0
    if len(line) < tgt:
        return None
    for p in line:
        if p == Piece.B:
            acc = 0
        else:
            if pre is None or p == pre:
                acc += 1
            else:
                acc = 1
        if acc >= tgt:
            return p
        pre = p
    return None


class GravityGoBang(GoBang):
    def get_valid_actions(self, state: State) -> List[Action]:
        def _get_valid_r(col):
            for idx, p in enumerate(col):
                if p != Piece.B:
                    return idx - 1

        actions = []
        board = state.get_board()
        for idx, col in enumerate(board.gen_cols()):
            r = _get_valid_r(col)
            if 0 <= r < self.board_size:
                action = Action(idx=idx, r=r, c=idx)
                actions.append(action)
        return actions


TICTACTOE = GoBang(board_size=3, target_size=3, name="TicTacToe")
GGOBANG4 = GravityGoBang(board_size=4, target_size=3, name="GGoBang4")
GGOBANG = GravityGoBang(board_size=7, target_size=4, name="GGoBang")

