#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     go.py
   Author :       chenhao
   time：          2022/3/15 15:19
   Description :
-------------------------------------------------
"""
import copy
import logging
import types
from collections import defaultdict
from typing import List, Type

from rlb.core import Board, Piece, BoardEnv, State, Cord, CordPiece
from rlb.core import TransferInfo
from rlb.gobang import BoardAction

logger = logging.getLogger(__name__)


class GogAction(BoardAction):
    pass


def get_neighbors(board: Board, cord: Cord) -> List[CordPiece]:
    rs = []
    for r in [cord.r - 1, cord.r + 1]:
        if 0 <= r < board.row_num:
            rs.append(CordPiece(r=r, c=cord.c, piece=board.get_piece(r, cord.c)))

    for c in [cord.c - 1, cord.c + 1]:
        if 0 <= c < board.col_num:
            rs.append(CordPiece(r=cord.r, c=c, piece=board.get_piece(cord.r, c)))

    return rs


class Block:
    def __init__(self, cord_pieces: List[CordPiece], neighbors: List[CordPiece]):
        self.cord_pieces = sorted(cord_pieces)
        self.piece = cord_pieces[0].piece
        self.neighbors = sorted(neighbors)
        self.liberties: List[CordPiece] = [e for e in self.neighbors if e.piece == Piece.B]
        self.liberty_num = len(self.liberties)
        self.is_alive = self.liberty_num > 0

    def __repr__(self):
        return f"pieces:[{','.join([str(e) for e in self.cord_pieces])}], liberties:[{','.join([str(e) for e in self.liberties])}]"

    def size(self):
        return len(self.cord_pieces)

    def is_kill_by(self, cord_piece: CordPiece) -> bool:
        if cord_piece.piece in [Piece.B, self.piece]:
            return False
        if self.liberty_num > 1:
            return False
        return self.liberties[0].tuple() == cord_piece.tuple()


def get_block(board: Board, cord_piece: CordPiece) -> Block:
    assert cord_piece.piece != Piece.B
    q = [cord_piece]
    cord_pieces = set()
    neighbors = set()

    while q:
        cp = q.pop(0)
        cord_pieces.add(cp)
        nbs = get_neighbors(board, cp)
        for nb in nbs:
            if nb.piece == cord_piece.piece:
                if nb not in cord_pieces:
                    q.append(nb)
            else:
                neighbors.add(nb)
    block = Block(cord_pieces=list(cord_pieces), neighbors=list(neighbors))
    return block


def board2blocks(board: Board) -> List[Block]:
    blocks = []
    visited = set()

    for r, row in enumerate(board.rows):
        for c, p in enumerate(row):
            if p == Piece.B:
                continue
            cord_piece = CordPiece(r=r, c=c, piece=p)
            if cord_piece in visited:
                continue
            block = get_block(board, cord_piece)
            visited |= set(block.cord_pieces)
            blocks.append(block)
    return blocks


def remove_block(board: Board, block: Block):
    logger.debug(f"removing block:{block}")

    for cp in block.cord_pieces:
        board.set_piece(cp.r, cp.c, Piece.B)


class GoState(State):
    last_dead_cord_piece: CordPiece = None


class BaseGo(BoardEnv):
    draw2loss: bool = False
    pieces: list = [Piece.X, Piece.O]

    @classmethod
    def _get_kill_cord_piece(cls, block: Block) -> CordPiece:
        if block.liberty_num != 1:
            return None
        cp = copy.copy(block.liberties[0])
        cp.piece = cls._get_next_piece(block.piece)
        return cp

    def _reset(self):
        board = Board(row_num=self.board_size, col_num=self.board_size)
        self._state = GoState(board=board, piece=Piece.X, pass_num=0)

    @classmethod
    def get_valid_actions_by_state(cls, state: GoState) -> List[GogAction]:
        actions = []
        board, piece = copy.copy(state.board), state.piece
        blocks = board2blocks(board=state.board)

        cand_kill_map = defaultdict(list)
        for block in blocks:
            kill_cp = cls._get_kill_cord_piece(block)
            if kill_cp:
                cand_kill_map[kill_cp].append(block)

        for r in range(cls.board_size):
            for c in range(cls.board_size):
                if board.get_piece(r, c) != Piece.B:
                    continue
                cord_piece = CordPiece(r=r, c=c, piece=state.piece)
                if cord_piece in cand_kill_map:
                    kill_blocks = cand_kill_map[cord_piece]
                    if cord_piece == state.last_dead_cord_piece and len(kill_blocks) == 1 \
                            and kill_blocks[0].size() == 1:
                        pass
                    else:
                        actions.append(cls.action_cls(r=r, c=c))
                else:
                    board.set_piece(row=r, col=c, piece=state.piece)
                    block = get_block(board, cord_piece)
                    if block.is_alive:
                        actions.append(cls.action_cls(r=r, c=c))
                    board.set_piece(row=r, col=c, piece=Piece.B)

        return actions

    @classmethod
    def _on_pass_action(cls, state: GoState) -> TransferInfo:
        next_state = copy.deepcopy(state)
        next_state.piece = cls._get_next_piece(state.piece)
        transfer_info = TransferInfo(next_state=next_state, reward=0., is_done=False, extra_info=dict(is_pass=True))
        return transfer_info

    @classmethod
    def _on_valid_action(cls, state: GoState, action: GogAction) -> TransferInfo:
        r, c = action.r, action.c
        next_state = copy.deepcopy(state)
        next_state.last_dead_cord_piece = None
        board, piece = next_state.board, state.piece
        cord_piece = CordPiece(r=r, c=c, piece=piece)

        blocks = board2blocks(board=state.board)
        for block in blocks:
            if block.is_kill_by(cord_piece):
                remove_block(board, block)
                if block.size() == 1:
                    next_state.last_dead_cord_piece = block.cord_pieces[0]

        board.set_piece(row=r, col=c, piece=piece)
        next_state.piece = cls._get_next_piece(piece)
        is_done = False
        extra_info = dict()
        return TransferInfo(next_state=next_state, reward=0., is_done=is_done, extra_info=extra_info)

    @classmethod
    def _on_invalid_action(cls, state: GoState, action: GogAction) -> TransferInfo:
        logger.debug(f"invalid action:{action}, equal to pass")
        return cls._on_pass_action(state)


def build_go_cls(name, board_size: int) -> Type[BaseGo]:
    action_num = board_size ** 2 + 1
    action_cls = types.new_class(name=f"{name}Action", bases=(GogAction,),
                                 kwds={}, exec_body=lambda x: x.update(_board_size=board_size))

    attrs = dict(board_size=board_size, action_num=action_num, action_cls=action_cls)
    cls = types.new_class(name=name, bases=(BaseGo,), kwds={}, exec_body=lambda x: x.update(**attrs))
    cls.__module__ = __name__
    return cls


Go4 = build_go_cls(name="Go4", board_size=4)

__all__env_cls__ = [Go4]

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    board_str = """
_OX_
XX__
____
____
    """
    env = Go4
    action_cls = Go4.action_cls
    board = Board.from_str(board_str)
    piece = Piece.O

    state = GoState(board=board, piece=piece)
    logger.info(state.render_str())

    action = action_cls(r=0, c=0, piece=piece)
    transfer_info = env.transfer(state=state, action=action)
    logger.info(transfer_info)
    next_state = transfer_info.next_state
    logger.info(next_state.render_str())
