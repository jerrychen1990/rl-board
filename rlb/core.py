#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     core.py
   Author :       chenhao
   time：          2022/4/6 18:05
   Description :
-------------------------------------------------
"""
import json
import logging
import os
from abc import abstractmethod
from collections import namedtuple
from enum import Enum
from functools import lru_cache
from itertools import chain
from multiprocessing import Process
from typing import List, Optional, Type

import numpy as np
from pydantic import BaseModel
from snippets import flat, groupby

from rlb.utils import sample_by_probs

logger = logging.getLogger(__name__)


class Piece(int, Enum):
    B = 0
    X = 1
    O = -1

    # def __str__(self):
    #     s = repr(self)
    #     return "_" if s == "B" else s

    @classmethod
    def from_str(cls, s):
        return cls["B" if s == "_" else s]

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "_" if self.name == "B" else self.name


Cord = namedtuple("Cord", ["r", "c"])
CordP = namedtuple("CordP", ["r", "c", "p"])


class Board:
    def __init__(self, data: np.ndarray):
        self.r_size, self.c_size = data.shape
        self._data = data

    def set_piece(self, r: int, c: int, p: Piece):
        self._data[r][c] = p

    def get_piece(self, r, c) -> Piece:
        return Piece(self._data[r][c])

    def get_cordp(self, r, c) -> CordP:
        p = self.get_piece(r, c)
        return CordP(r, c, p)

    def get_col(self, c) -> List[Piece]:
        return [Piece(e) for e in self._data[:, c]]

    def get_row(self, r) -> List[Piece]:
        return [Piece(e) for e in self._data[r]]

    def gen_rows(self, return_cordp=False):
        for i in range(self.r_size):
            if return_cordp:
                yield [CordP(r=i, c=j, p=p) for j, p in enumerate(self.get_row(i))]
            else:
                yield self.get_row(i)

    def gen_cols(self, return_cordp=False):
        for i in range(self.c_size):
            if return_cordp:
                yield [CordP(r=j, c=i, p=p) for j, p in enumerate(self.get_col(i))]
            else:
                yield self.get_col(i)

    def gen_rd_slashes(self, return_cordp=False):
        b, r, c = self._data, self.r_size, self.c_size

        for idx in range(1, r + c):
            if return_cordp:
                yield [self.get_cordp(r - idx + i, i) for i in range(max(0, idx - r), min(c, idx))]
            else:
                yield [self.get_piece(r - idx + i, i) for i in range(max(0, idx - r), min(c, idx))]

    def gen_ru_slashes(self, return_cordp=False):
        b, r, c = self._data, self.r_size, self.c_size

        for idx in range(1, r + c):
            if return_cordp:
                yield [self.get_cordp(idx - 1 - i, i) for i in range(max(0, idx - r), min(c, idx))]
            else:
                yield [self.get_piece(idx - 1 - i, i) for i in range(max(0, idx - r), min(c, idx))]

    def gen_all_lines(self, return_cordp=False):
        return chain(self.gen_rows(return_cordp), self.gen_cols(return_cordp), self.gen_rd_slashes(return_cordp),
                     self.gen_ru_slashes(return_cordp))

    def gen_line_with_dir(self, r, c, dr, dc, include_beg=True, return_cordp=False):
        if include_beg:
            yield self.get_cordp(r, c) if return_cordp else self.get_piece(r, c)
        r += dr
        c += dc
        while 0 <= r < self.r_size and 0 <= c < self.c_size:
            yield self.get_cordp(r, c) if return_cordp else self.get_piece(r, c)
            r += dr
            c += dc

    def count_pieces(self) -> dict:
        pieces = flat(self.gen_rows())
        return groupby(pieces)

    def is_full(self):
        return (self._data != Piece.B).all()

    def is_empty(self):
        return (self._data == Piece.B).all()

    def __repr__(self):
        return "\n".join("".join(str(Piece(e)) for e in r) for r in self.gen_rows())

    def __eq__(self, other):
        return isinstance(other, Board) and hash(other) == hash(self)

    def __hash__(self):
        return hash(str(self))

    def get_immutable_data(self):
        self._data.flags.writeable = False
        return self._data

    @classmethod
    def get_empty_board(cls, r, c=None):
        if c is None:
            c = r
        board = np.zeros(shape=(r, c)).astype(np.int8)
        return Board(data=board)

    @classmethod
    def from_str(cls, s):
        b = np.array([[Piece.from_str(e) for e in r] for r in s.strip().split("\n")])
        return Board(b)


class State:
    def __init__(self, data: np.ndarray, piece: Piece):
        assert data.flags.writeable == False
        self._data: np.ndarray = data
        self._piece = piece

    def __eq__(self, other):
        return isinstance(other, State) and hash(other) == hash(self)

    def __hash__(self):
        return hash((self._data.tobytes(), self._piece))

    @property
    def data(self):
        return self._data

    @property
    def piece(self):
        return self._piece

    @classmethod
    @lru_cache(maxsize=5000)
    def from_board_piece(cls, board: Board, piece: Piece):
        # logger.info("exe")
        return State(board.get_immutable_data(), piece)

    @classmethod
    def from_str(cls, s):
        board_s, p_s = s.split("|")
        b = Board.from_str(board_s.replace(",", "\n"))
        p = Piece.from_str(p_s.strip())
        return cls.from_board_piece(b, p)

    def get_board(self):
        return Board(data=self._data)

    def to_new_board(self):
        data = self._data.copy()
        data.flags.writeable = True
        return Board(data=data)

    def __repr__(self):
        return ",".join("".join(str(Piece(e)) for e in r) for r in self._data) + "|" + str(self._piece)


class Action(BaseModel):
    idx: int
    r: int
    c: int
    is_pass = False

    def __repr__(self):
        if self.is_pass:
            return "Pass"
        return f"[{self.r},{self.c}]"

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash((self.idx, self.r, self.c, self.is_pass))

    def __eq__(self, other):
        return isinstance(other, Action) and hash(other) == hash(self)


PASS_ACTION = Action(idx=0, r=0, c=0, is_pass=True)


class Edge(BaseModel):
    reward: float
    is_done: bool
    win_piece: Optional[Piece]


class TransferInfo(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    next_state: State
    reward: float
    is_done: bool
    win_piece: Optional[Piece]
    extra_info: dict


class ActionInfo(BaseModel):
    action_idx: int
    prob: float
    probs: List[float]


def action_info2action(action_info: ActionInfo, board_size: int) -> Action:
    r, c = divmod(action_info.action_idx, board_size)
    is_pass = r == c == board_size
    return Action(idx=action_info.action_idx, r=r, c=c, is_pass=is_pass)


class BoardEnv:
    pieces = NotImplemented
    allow_pass = NotImplemented

    def __init__(self, name: str, board_size: int):
        self.name = name
        self.board_size = board_size
        self._next_piece_dict = dict(zip(self.pieces, self.pieces[1:] + self.pieces[0:1]))
        self.action_num = self.board_size * self.board_size
        if self.allow_pass:
            self.action_num += 1

    def get_next_piece(self, piece: Piece) -> Piece:
        return self._next_piece_dict[piece]

    @abstractmethod
    def reset(self) -> State:
        raise NotImplementedError

    @abstractmethod
    def judge_state(self, state: State) -> Edge:
        raise NotImplementedError

    @abstractmethod
    def get_valid_actions(self, state: State) -> List[Action]:
        raise NotImplementedError

    def build_action(self, action_info: ActionInfo) -> Action:
        return action_info2action(action_info=action_info, board_size=self.board_size)

    def get_mask(self, state: State) -> List[float]:
        mask = [0.] * self.action_num
        valid_actions = self.get_valid_actions(state)
        for action in valid_actions:
            mask[action.idx] = 1.
        return mask

    @abstractmethod
    def on_valid_action(self, state: State, action: Action) -> TransferInfo:
        raise NotImplementedError

    def on_pass_action(self, state: State) -> TransferInfo:
        next_state = State(state.data, self.get_next_piece(state.piece))
        edge = self.judge_state(next_state)
        transfer_info = TransferInfo(next_state=next_state, **edge.dict(), extra_info=dict(is_pass=True))
        return transfer_info

    def transfer(self, state: State, action: Action) -> TransferInfo:
        if action.is_pass:
            return self._on_pass_action(state)
        else:
            return self.on_valid_action(state=state, action=action)

    @classmethod
    def render(cls, state: State):
        logger.info("\n" + str(state).replace(",", "\n"))

    def close(self):
        pass

    def seed(self, seed):
        pass


class Agent:
    def __init__(self, name, action_num, *args, **kwargs):
        self.name = name
        self.action_num = action_num

    def get_weights(self, state: State, mode: str, **kwargs) -> List[float]:
        raise NotImplementedError

    def choose_action(self, state: State, mode: str, mask: List[int], **kwargs) -> ActionInfo:
        sum_mask = sum(mask)
        assert sum_mask > 0
        weights = self.get_weights(state=state, mode=mode, **kwargs)

        assert len(weights) == len(mask)
        assert min(weights) >= 0
        weights = [w * m for w, m in zip(weights, mask)]
        sum_weights = sum(weights)
        if sum_weights == 0:
            probs = [m / sum_mask for m in mask]
        else:
            probs = [w / sum_weights for w in weights]
        action_idx, prob = sample_by_probs(probs=probs)
        # logger.debug(f"choose action:{action_idx} with prob:{prob:2.3f}")
        action_info = ActionInfo(action_idx=action_idx, prob=prob, probs=probs)
        return action_info

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"


class RandomAgent(Agent):
    def get_weights(self, state: State, mode: str, **kwargs) -> List[float]:
        return [1 / self.action_num] * self.action_num


class Step(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    agent_name: str
    state: State
    action: Action
    prob: float
    probs: List[float]
    next_state: State
    reward: float
    is_done: bool
    win_piece: Optional[Piece]
    extra_info: dict = {}

    @property
    def cur_piece(self) -> Piece:
        return self.state.piece

    @property
    def next_piece(self) -> Piece:
        return self.next_state.piece


class ACInfo(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    state: State
    value: float
    probs: Optional[List[float]]

    def to_json_dict(self):
        d = self.dict(exclude_none=True)
        d["state"] = str(self.state)
        return d

    @classmethod
    def from_json_dict(cls, d):
        d["state"] = State.from_str(d["state"])
        return ACInfo(**d)

    def detail_str(self, action_cls: Type[Action] = None) -> str:
        rs = self.state.render_str()
        p_infos = []
        if self.probs:
            for idx, p in sorted(enumerate(self.probs), key=lambda x: x[1], reverse=True):
                p_info = f"{action_cls.from_idx(idx)}:{p:2.3f}" if action_cls else f"{idx}:{p:2.3f}"
                p_infos.append(p_info)
        rs += "\n" + f"probs:[{', '.join(p_infos)}]"
        rs += "\n" + f"{self.value=:2.3f}"
        return rs


class Episode(BaseModel):
    steps: List[Step]
    cost: float
    win_piece: Optional[Piece]
    winner: Optional[str]

    def __len__(self) -> int:
        return len(self.steps)

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

    def __str__(self):
        return f"[step:{self.step:d}, cost:{self.cost:5.3f}s]"


class Context(BaseModel):
    base_dir: str

    @property
    def model_dir(self):
        return os.path.join(self.base_dir, "models")

    @property
    def best_model_path(self):
        return os.path.join(self.model_dir, "best_model.pt")

    @property
    def record_dir(self):
        return os.path.join(self.base_dir, "records")

    def ckpt_model_path(self, ckpt: int):
        return os.path.join(self.model_dir, f"model-{ckpt}.pt")

    @property
    def config_path(self):
        return os.path.join(self.base_dir, "config.json")


class BaseProcess(Process):
    def __init__(self, context: Context, run_kwargs=dict(), *args, **kwargs):
        super(BaseProcess, self).__init__(*args, **kwargs)
        self.context = context
        self.run_kwargs = run_kwargs
