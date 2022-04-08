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
from functools import lru_cache
from typing import List, Type, Set

from rlb.core import TransferInfo, Board, Piece, BoardEnv, State, CordP, Cord, Action, Edge

logger = logging.getLogger(__name__)


class Othello(BoardEnv):
    draw2loss: bool = False
    pieces: list = [Piece.X, Piece.O]

    def reset(self) -> State:
        assert self.board_size % 2 == 0
        mid = self.board_size // 2
        x, o = self.pieces[0], self.pieces[1]
        board = Board.get_empty_board(self.board_size, self.board_size)
        board.set_piece(mid - 1, mid - 1, x)
        board.set_piece(mid, mid, x)
        board.set_piece(mid - 1, mid, o)
        board.set_piece(mid, mid - 1, o)
        return State.from_board_piece(board, self.pieces[0])

    @lru_cache(maxsize=5000)
    def get_valid_cordps_by_board(self, board: Board) -> List[CordP]:
        cps = set()
        for line in board.gen_all_lines(return_cordp=True):
            cps |= get_cordp_by_line(line)
        return list(sorted(cps))

    def judge_state(self, state: State) -> Edge:
        board = state.get_board()
        is_done = not self.get_valid_cordps_by_board(board)
        win_piece = None
        reward = 0.
        if is_done:
            piece_dict = board.count_pieces()
            max_cnt = max(piece_dict.values())
            win_pieces = [k for k, v in piece_dict.items() if v == max_cnt]
            win_piece = win_pieces[0] if len(win_pieces) == 1 else None
        return Edge(win_piece=win_piece, is_done=is_done, reward=reward)

    def get_valid_actions(self, state: State) -> List[Action]:
        board, piece = state.get_board(), state.piece
        cordps = self.get_valid_cordps_by_board(board)
        actions = [Action(r=cp.r, c=cp.c, idx=cp.r * self.board_size + cp.c) for cp in cordps if cp.p == piece]
        return actions

    def on_valid_action(self, state: State, action: Action) -> TransferInfo:
        piece = state.piece
        r, c = action.r, action.c
        next_board = state.to_new_board()
        next_piece = self.get_next_piece(piece)

        assert next_board.get_piece(r, c) == Piece.B
        change_cordps = set()
        for dr in [0, 1, -1]:
            for dc in [0, 1, -1]:
                if dr == dc == 0:
                    continue
                tmp = set()
                for cp in next_board.gen_line_with_dir(r, c, dr, dc, include_beg=False, return_cordp=True):
                    if cp.p == next_piece:
                        tmp.add(cp)
                    else:
                        if cp.p == piece:
                            change_cordps |= tmp
                        break

        assert change_cordps
        change_cordps.add(Cord(r=r, c=c))

        for cordp in change_cordps:
            next_board.set_piece(cordp.r, cordp.c, piece)
        next_state = State.from_board_piece(next_board, next_piece)
        edge = self.judge_state(next_state)
        return TransferInfo(next_state=next_state, **edge.dict(),extra_info={})


def get_cordp_by_line(line: List[CordP]) -> Set[CordP]:
    pre = None
    start = None
    pairs = []
    cordps = set()
    for cp in line:
        if pre and cp.p != pre.p:
            if start:
                pairs.append((start, cp))
            start = pre if cp.p != Piece.B else None
        pre = cp

    for s, e in pairs:
        if s and e:
            if s.p == Piece.B and e.p != Piece.B:
                cp = CordP(s.r, s.c, e.p)
                cordps.add(cp)
            if e.p == Piece.B and s.p != Piece.B:
                cp = CordP(e.r, e.c, s.p)
                cordps.add(cp)
    return cordps


OTHELLO = Othello(name="othello", board_size=8)
OTHELLO4 = Othello(name="othello4", board_size=4)

if __name__ == "__main__":
    line = "__XXO_OOXOO_OO"
    line = [CordP(r=0, c=c, p=Piece.from_str(s)) for c, s in enumerate(line)]
    for cp in line:
        logger.info(cp)
    logger.info(sorted(get_cordp_by_line(line)))
