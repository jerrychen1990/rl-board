#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     othello.py
   Author :       chenhao
   time：          2022/4/2 16:13
   Description :
-------------------------------------------------
"""
import copy
import logging
import types
from collections import defaultdict
from typing import List, Type, Tuple

from pydantic import BaseModel
from snippets import flat

from rlb.board_core import Board, Piece, BoardEnv, BoardState, CordPiece, Cord
from rlb.core import Action, TransferInfo, State
from rlb.gobang import BoardAction

logger = logging.getLogger(__name__)


class OthelloAction(BoardAction):
    pass


class BaseOthello(BoardEnv):
    draw2loss: bool = False
    pieces: list = [Piece.X, Piece.O]

    @classmethod
    def get_valid_cord_pieces_by_line(cls, line: List[CordPiece]) -> List[CordPiece]:
        idx = 0
        contain_piece = set()
        cord_pieces = []
        while idx < len(line) and line[idx].piece == Piece.BLANK:
            idx += 1
        if 0 < idx < len(line):
            cord_piece: CordPiece = copy.copy(line[idx - 1])
            cord_piece.piece = cls._get_next_piece(line[idx].piece)
            cord_pieces.append(cord_piece)

        while idx < len(line) and line[idx].piece != Piece.BLANK:
            contain_piece.add(line[idx].piece)
            idx += 1
        if idx < len(line):
            cord_piece: CordPiece = copy.copy(line[idx])
            cord_piece.piece = cls._get_next_piece(line[idx - 1].piece)
            cord_pieces.append(cord_piece)
        cord_pieces = [e for e in cord_pieces if e.piece in contain_piece]
        return cord_pieces

    def _reset(self):
        assert self.board_size % 2 == 0
        mid = self.board_size // 2
        x, o = self.pieces[0], self.pieces[1]
        board = Board(row_num=self.board_size, col_num=self.board_size)
        board.set_piece(mid - 1, mid - 1, x)
        board.set_piece(mid, mid, x)
        board.set_piece(mid - 1, mid, o)
        board.set_piece(mid, mid - 1, o)

        self._state = BoardState(board=board, piece=Piece.X, pass_num=0)

    @classmethod
    def _on_pass_action(cls, state: BoardState) -> TransferInfo:
        logger.debug("on pass")
        next_state = copy.deepcopy(state)
        next_state.piece = cls._get_next_piece(state.piece)
        transfer_info = TransferInfo(next_state=next_state, reward=0., is_done=False, extra_info=dict())
        return transfer_info

    @classmethod
    def _on_invalid_action(cls, state: BoardState, action: OthelloAction) -> TransferInfo:
        logger.debug(f"invalid action :{action}, will pass")

        return cls._on_pass_action(state)
        next_state = copy.deepcopy(state)
        transfer_info = TransferInfo(next_state=next_state, reward=0., is_done=False, extra_info=dict(valid=False))
        return transfer_info

    @classmethod
    def _on_valid_action(cls, state: BoardState, action: OthelloAction) -> TransferInfo:
        next_state = copy.deepcopy(state)
        board, piece = next_state.board, next_state.piece
        r, c = action.r, action.c
        assert board.get_piece(r, c) == Piece.BLANK
        change_cords = set()
        op_piece = cls._get_next_piece(piece)

        for dr in [0, 1, -1]:
            for dc in [0, 1, -1]:
                if dr == dc == 0:
                    continue
                tmp = set()
                for cp in board.gen_cp_iterator(r, c, dr, dc, False):
                    if cp.piece == op_piece:
                        tmp.add(Cord(**cp.dict()))
                    else:
                        if cp.piece == piece:
                            change_cords |= tmp
                        break

        assert change_cords
        change_cords.add(Cord(r=r, c=c))

        logger.info(change_cords)
        for cord in change_cords:
            board.set_piece(cord.r, cord.c, piece)
        next_state.piece = op_piece
        extra_info = dict()
        is_done = board.is_full()
        if is_done:
            piece_dict = board.count_pieces()
            extra_info["win_piece"] = piece_dict.keys()[0]
        return TransferInfo(next_state=next_state, reward=0., is_done=is_done, extra_info=extra_info)

    @classmethod
    def get_valid_actions_by_state(cls, state: BoardState) -> List[OthelloAction]:
        board, piece = state.board, state.piece
        lines = board.get_lines()
        valid_actions = set()
        for line in lines:
            cord_pieces = cls.get_valid_cord_pieces_by_line(line)
            for cord_piece in cord_pieces:
                if cord_piece.piece == piece:
                    valid_actions.add(cls.action_cls(r=cord_piece.r, c=cord_piece.c))

        return list(valid_actions)


def build_othello_cls(name, board_size: int) -> Type[BaseOthello]:
    action_num = board_size ** 2
    action_cls = types.new_class(name=f"{name}Action", bases=(BoardAction,),
                                 kwds={}, exec_body=lambda x: x.update(_board_size=board_size))

    attrs = dict(board_size=board_size, action_num=action_num, action_cls=action_cls)
    cls = types.new_class(name=name, bases=(BaseOthello,), kwds={}, exec_body=lambda x: x.update(**attrs))
    cls.__module__ = __name__
    return cls


Othello4 = build_othello_cls(name="Othello4", board_size=4)
Othello8 = build_othello_cls(name="Othello8", board_size=8)
Othello = build_othello_cls(name="Othello", board_size=8)

__all__env_cls__ = [Othello4, Othello8, Othello]
