#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     board_core.py
   Author :       chenhao
   time：          2022/3/16 15:55
   Description :
-------------------------------------------------
"""
from __future__ import annotations

import copy
import logging
from abc import abstractmethod
from enum import Enum
from functools import lru_cache
from typing import Tuple, List, Optional, Type

from pydantic import BaseModel, Field
from snippets import groupby, flat

from rlb.core import Action, State, ActionInfo, TransferInfo, Env
from rlb.utils import tuplize

logger = logging.getLogger(__name__)


class Piece(str, Enum):
    BLANK = "_"
    X = "X"
    O = "O"

    def __repr__(self):
        return self.value


class Cord(BaseModel):
    r: int
    c: int

    def tuple(self):
        return self.r, self.c

    def __hash__(self):
        return hash(self.tuple())

    def __eq__(self, other):
        return isinstance(other, Cord) and hash(other) == hash(self)

    def __str__(self):
        return f"[{self.r},{self.c}]"

    def __lt__(self, other):
        return self.tuple() < other.tuple()


class CordPiece(Cord):
    piece: Piece

    def __str__(self):
        return f"[{self.r},{self.c}:{self.piece.value}]"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, CordPiece) and hash(other) == hash(self)


class Board(BaseModel):
    row_num: int
    col_num: int
    board: List[List[Piece]] = None

    def __init__(self, **data):
        super(Board, self).__init__(**data)
        if self.board is None:
            self.board = [[Piece.BLANK] * self.col_num for _ in range(self.row_num)]
        else:
            assert len(self.board) == self.row_num
            assert len(self.board[0]) == self.col_num

    def set_piece(self, row, col, piece):
        self.board[row][col] = piece

    def get_piece(self, row, col):
        return self.board[row][col]

    def flatten(self):
        return "".join("".join(r) for r in self.board)

    def is_empty(self, row, col):
        return self.board[row][col] == Piece.BLANK

    def is_full(self):
        for r in self.board:
            for e in r:
                if e == Piece.BLANK:
                    return False
        return True

    def get_col(self, idx):
        return [e[idx] for e in self.board]

    @property
    def rows(self):
        return self.board

    @property
    def cols(self):
        return [[e[c] for e in self.board] for c in range(self.col_num)]

    @property
    def diagonals(self):
        assert self.row_num == self.col_num
        return zip(*[[self.board[i][i], self.board[i][self.col_num - 1 - i]] for i in range(self.row_num)])

    @property
    def slashes(self):
        board, row_num, col_num = self.board, self.row_num, self.col_num

        rd = [[board[row_num - l + i][i] for i in range(l) if 0 <= i < col_num and 0 <= row_num - l + i < row_num]
              for l in range(1, col_num + row_num)]
        ru = [[board[l - 1 - i][i] for i in range(l) if 0 <= i < col_num and 0 <= l - i - 1 < row_num]
              for l in range(1, col_num + row_num)]

        return rd + ru

    def get_cord_piece(self, r, c):
        return CordPiece(r=r, c=c, piece=self.board[r][c])

    def get_lines(self) -> List[List[CordPiece]]:
        board, row_num, col_num = self.board, self.row_num, self.col_num

        rows = [[self.get_cord_piece(r, c) for c in range(col_num)] for r in range(row_num)]
        cols = [[self.get_cord_piece(r, c) for r in range(row_num)] for c in range(col_num)]
        rd = [[self.get_cord_piece(row_num - l + i, i) for i in range(l) if
               0 <= i < col_num and 0 <= row_num - l + i < row_num]
              for l in range(1, col_num + row_num)]
        ru = [[self.get_cord_piece(l - 1 - i, i) for i in range(l) if 0 <= i < col_num and 0 <= l - i - 1 < row_num]
              for l in range(1, col_num + row_num)]
        return rows + cols + rd + ru

    def gen_cp_iterator(self, r, c, dr, dc, contain=True):
        if not contain:
            r += dr
            c += dc
        while 0 <= r < self.row_num and 0 <= c < self.col_num:
            yield CordPiece(r=r, c=c, piece=self.board[r][c])
            r += dr
            c += dc

    def __str__(self):
        return "\n".join("".join(r) for r in self.board)

    def to_tuple(self):
        return tuplize(self.board)

    def count_pieces(self):
        return groupby(flat(self.board))

    @classmethod
    def from_str(cls, s: str):
        b = [list(e) for e in s.strip().split("\n")]
        board_size = len(b)
        board = Board(row_num=board_size, col_num=board_size, board=b)
        return board


class BoardAction(Action):
    _board_size = NotImplemented
    r: int
    c: int

    def to_idx(self) -> int:
        if self.is_pass:
            return self._board_size * self._board_size
        return self.r * self._board_size + self.c

    @classmethod
    def from_idx(cls, idx):
        assert idx <= cls._board_size * cls._board_size
        if idx == cls._board_size * cls._board_size:
            return cls(r=0, c=0, is_pass=True)

        r, c = divmod(idx, cls._board_size)
        return cls(r=r, c=c)

    @classmethod
    def from_cmd(cls, cmd: str) -> int:
        if cmd.upper() == "P":
            action = cls(is_pass=True, r=0, c=0)
        else:
            r, c = cmd.split(",")
            r, c = int(r.strip()), int(c.strip())
            action = cls(r=r, c=c)
        return action.to_idx()

    @classmethod
    def get_pass_action(cls):
        return cls(r=0, c=0, is_pass=True)

    def __str__(self):
        return f"[{self.r},{self.c}]" if not self.is_pass else "pass"


class BoardState(State):
    board: Board
    piece: Piece
    pass_num = 0

    def to_tuple(self) -> Tuple:
        return self.board.to_tuple(), self.piece, self.pass_num

    def render_str(self) -> str:
        rs = f"\n{self.board}|{self.piece}"
        if self.pass_num > 0:
            rs += f"|{self.pass_num}"
        return rs

    @classmethod
    def from_render_str(cls, s: str):
        board_str, piece_str, *args = s.strip().split("|")
        board = Board.from_str(board_str)
        piece = Piece(piece_str)
        pass_num = args[0] if args else 0
        return cls(board=board, piece=piece, pass_num=pass_num)


class Step(BaseModel):
    agent_name: str
    state: BoardState
    action_idx: int
    prob: float = Field(le=1., ge=0.)
    probs: List[float]
    reward: float
    next_state: BoardState
    is_done: bool
    extra_info: dict = {}

    @property
    def cur_piece(self) -> Piece:
        return self.state.piece

    @property
    def next_piece(self) -> Piece:
        return self.next_state.piece

    @classmethod
    def from_info(cls, agent_name, action_info: ActionInfo, transfer_info: TransferInfo):
        return cls(agent_name=agent_name, **action_info.dict(), **transfer_info.dict())


class ACInfo(BaseModel):
    state: BoardState
    value: float
    probs: Optional[List[float]]
    is_in_path: bool = True

    def detail_str(self, action_cls: Type[Action] = None) -> str:
        rs = self.state.render_str()
        p_infos = []
        if self.probs:
            for idx, p in sorted(enumerate(self.probs), key=lambda x: x[1], reverse=True):
                p_info = f"{action_cls.from_idx(idx)}:{p:2.3f}" if action_cls else f"{idx}:{p:2.3f}"
                p_infos.append(p_info)
        rs += "\n" + f"probs:[{', '.join(p_infos)}]"
        rs += "\n" + f"{self.value=:2.3f}, {self.is_in_path=}"
        return rs


class Episode(BaseModel):
    steps: List[Step] = list()
    cost: float = 0.
    win_piece: Optional[Piece]
    winner: Optional[str]

    def to_ac_infos(self, value_decay=1.) -> List[ACInfo]:
        ac_infos = []
        origin_value = 1 if self.win_piece else 0
        value = origin_value

        for step in self.steps[::-1]:
            value *= value_decay
            eff = 1 if step.cur_piece == self.win_piece else -1
            ac_infos.append(ACInfo(state=step.state, value=value * eff, probs=step.probs))
        ac_infos.reverse()
        last_step = self.steps[-1]
        ac_infos.append(ACInfo(state=last_step.next_state,
                               value=origin_value if last_step.next_piece == self.win_piece else -origin_value))
        return ac_infos

    @property
    def step(self):
        return len(self.steps)

    def __str__(self):
        return f"[step:{self.step:d}, cost:{self.cost:5.3f}s]"


class BoardEnv(Env):
    board_size: int = NotImplemented
    draw2loss: bool = NotImplemented
    pieces: list = NotImplemented

    def __init__(self):
        self._reset()

    def _reset(self):
        board = Board(row_num=self.board_size, col_num=self.board_size)
        self._state = BoardState(board=board, piece=Piece.X, pass_num=0)

    def get_state(self) -> BoardState:
        return copy.copy(self._state)

    def reset(self) -> BoardState:
        self._reset()
        return self.get_state()

    def _set_state(self, state: BoardState):
        self._state = state

    @classmethod
    @lru_cache(maxsize=None)
    def _get_next_piece(cls, piece: Piece) -> Piece:
        idx = cls.pieces.index(piece)
        idx = (idx + 1) % len(cls.pieces)
        return cls.pieces[idx]

    def _change_piece(self):
        self._state.piece = self._get_next_piece(self._state.piece)

    @classmethod
    @abstractmethod
    def _on_pass_action(cls, state: BoardState) -> TransferInfo:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _on_invalid_action(cls, state: BoardState, action: Action) -> TransferInfo:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _on_valid_action(cls, state: BoardState, action: Action) -> TransferInfo:
        raise NotImplementedError

    @classmethod
    def transfer(cls, state: BoardState, action: Action) -> TransferInfo:
        if action.is_pass:
            transfer_info = cls._on_pass_action(state)
        else:
            valid_actions = cls.get_valid_actions_by_state(state=state)
            # logger.info(f"{valid_actions=}")
            if action not in valid_actions:
                transfer_info = cls._on_invalid_action(state, action)
            else:
                transfer_info = cls._on_valid_action(state, action)
        return transfer_info

    def step(self, action: Action) -> TransferInfo:
        transfer_info = self.transfer(state=self._state, action=action)
        self._set_state(transfer_info.next_state)
        return transfer_info

    def render(self, mode="human"):
        logger.info(self._state.render_str())

    def close(self):
        pass

    def seed(self, seed):
        pass
