#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     core.py
   Author :       chenhao
   time：          2022/2/15 10:34
   Description :
-------------------------------------------------
"""
from __future__ import annotations
import logging
import os
from abc import abstractmethod, ABC, ABCMeta
from enum import Enum
from multiprocessing import Process
from typing import Tuple, List, Optional, Type
from functools import lru_cache

from pydantic import BaseModel, Field, validator

from rlb.utils import sample_by_probs, tuplize

logger = logging.getLogger(__name__)


class Action(ABC, BaseModel):
    @abstractmethod
    def to_idx(self) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_idx(cls, idx: int) -> Action:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_cmd(cls, cmd: str) -> Action:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        pass

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, Action) and hash(self) == hash(other)


class TransferInfo(BaseModel):
    next_obs: Tuple
    reward: float
    is_done: bool
    extra_info: dict


class State(metaclass=ABCMeta):
    @abstractmethod
    def obs(self) -> Tuple:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_obs(cls, obs: Tuple):
        raise NotImplementedError

    @abstractmethod
    def render_str(self) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_render_str(cls, render_str: str):
        raise NotImplementedError

    def __str__(self):
        return self.render_str()


class Env(metaclass=ABCMeta):
    action_cls: Type[Action] = NotImplemented
    action_num: int = NotImplemented

    @abstractmethod
    def get_obs(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def transfer(cls, obs: Tuple, action: Action) -> TransferInfo:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> TransferInfo:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_valid_actions_by_obs(cls, obs: Tuple) -> List[Action]:
        raise NotImplementedError

    @abstractmethod
    def get_valid_actions(self) -> List[Action]:
        obs = self.get_obs()
        return self.get_valid_actions_by_obs(obs=obs)

    @abstractmethod
    def reset(self) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def render(self, mode="human"):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def seed(self, seed):
        raise NotImplementedError


class ActionInfo(BaseModel):
    action_idx: int
    prob: float
    probs: List[float]


class Agent(ABC):
    def __init__(self, name, action_num, *args, **kwargs):
        self._name = name
        self.action_num = action_num

    @property
    def name(self):
        return self._name

    def get_weights(self, obs, mode, **kwargs) -> List[float]:
        raise NotImplementedError

    def choose_action(self, obs: Tuple, mode: str, **kwargs) -> ActionInfo:
        weights = self.get_weights(obs=obs, mode=mode, **kwargs)
        assert min(weights) >= 0
        sum_weights = sum(weights)
        if sum_weights == 0:
            probs = [1 / self.action_num] * self.action_num
        else:
            probs = [w / sum_weights for w in weights]
        action_idx, prob = sample_by_probs(probs=probs)
        # logger.debug(f"choose action:{action_idx} with prob:{prob:2.3f}")
        action_info = ActionInfo(action_idx=action_idx, prob=prob, probs=probs)
        return action_info

    def __repr__(self):
        return self.name

    def on_episode_end(self, episode_idx: int):
        pass


class RandomAgent(Agent):
    def get_weights(self, **kwargs) -> List[float]:
        return [1 / self.action_num] * self.action_num


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


class Piece(str, Enum):
    BLANK = "_"
    X = "X"
    O = "O"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class Step(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    agent_name: str
    obs: tuple
    action_idx: int
    prob: float = Field(le=1., ge=0.)
    probs: List[float]
    reward: float
    next_obs: Tuple
    is_done: bool
    extra_info: dict = {}

    @validator('obs')
    def convert2tuple(cls, v):
        return tuplize(v)

    @property
    def cur_piece(self) -> Piece:
        return self.obs[1]

    @property
    def next_piece(self) -> Piece:
        return self.next_obs[1]

    @classmethod
    def from_info(cls, agent_name, action_info: ActionInfo, transfer_info: TransferInfo):
        return cls(agent_name=agent_name, **action_info.dict(), **transfer_info.dict())


class ACInfo(BaseModel):
    obs: tuple
    value: float
    probs: Optional[List[float]]
    is_in_path: bool = True

    @validator('obs')
    def convert2tuple(cls, v):
        return tuplize(v)


class Episode(BaseModel):
    steps: List[Step] = list()
    cost: float = 0.
    win_piece: Optional[Piece]
    winner: Optional[str]

    def to_ac_info(self, value_decay=1.):
        ac_infos = []
        origin_value = 1 if self.win_piece else 0
        value = origin_value

        for step in self.steps[::-1]:
            value *= value_decay
            eff = 1 if step.cur_piece == self.win_piece else -1
            ac_infos.append(ACInfo(obs=step.obs, value=value * eff, probs=step.probs))
        ac_infos.reverse()
        last_step = self.steps[-1]
        ac_infos.append(ACInfo(obs=last_step.next_obs,
                               value=origin_value if last_step.next_piece == self.win_piece else -origin_value))
        return ac_infos

    @property
    def step(self):
        return len(self.steps)

    def __str__(self):
        return f"[step:{self.step:d}, cost:{self.cost:5.3f}s]"


class Board(object):
    def __init__(self, row_num, col_num, board=None):
        if board is None:
            self._board = [[Piece.BLANK] * col_num for _ in range(row_num)]
        else:
            self._board = board
        self.row_num = row_num
        self.col_num = col_num

    def set_piece(self, row, col, piece):
        self._board[row][col] = piece

    def get_piece(self, row, col):
        return self._board[row][col]

    def flatten(self):
        return "".join("".join(r) for r in self._board)

    def is_empty(self, row, col):
        return self._board[row][col] == Piece.BLANK

    def is_full(self):
        for r in self._board:
            for e in r:
                if e == Piece.BLANK:
                    return False
        return True

    def get_col(self, idx):
        return [e[idx] for e in self._board]

    @property
    def rows(self):
        return self._board

    @property
    def cols(self):
        return [[e[c] for e in self._board] for c in range(self.col_num)]

    @property
    def diagonals(self):
        assert self.row_num == self.col_num
        return zip(*[[self._board[i][i], self._board[i][self.col_num - 1 - i]] for i in range(self.row_num)])

    @property
    def slashes(self):
        board, row_num, col_num = self._board, self.row_num, self.col_num

        rd = [[board[row_num - l + i][i] for i in range(l) if 0 <= i < col_num and 0 <= row_num - l + i < row_num]
              for l in range(1, col_num + row_num)]
        ru = [[board[l - 1 - i][i] for i in range(l) if 0 <= i < col_num and 0 <= l - i - 1 < row_num]
              for l in range(1, col_num + row_num)]

        return rd + ru

    def __str__(self):
        return "\n".join("".join(r) for r in self._board)

    @classmethod
    def from_str(cls, s: str):
        b = [list(e) for e in s.strip().split("\n")]
        board_size = len(b)
        board = Board(row_num=board_size, col_num=board_size, board=b)
        return board


class InvalidActionStrategy(Enum):
    RETRY = "retry"
    PASS = "pass"
    FAIL = "fail"


class BoardState(State):

    def __init__(self, board: Board, next_piece: Piece):
        self.board = board
        self.next_piece = next_piece

    def obs(self) -> Tuple:
        return tuple(tuple(row) for row in self.board.rows), self.next_piece

    def render_str(self) -> str:
        return f"\n{self.board}|{self.next_piece}"

    @classmethod
    def from_render_str(cls, s: str):
        board_str, piece_str = s.strip().split("|")
        board = Board.from_str(board_str)
        piece = Piece(piece_str)
        return cls(board=board, next_piece=piece)

    @classmethod
    def from_obs(cls, obs: Tuple):
        b, next_piece = obs
        b = [list(e) for e in b]
        board_size = len(b)
        board = Board(row_num=board_size, col_num=board_size, board=b)
        return BoardState(board=board, next_piece=next_piece)


class BoardEnv(Env):
    board_size: int = NotImplemented
    draw2loss: bool = NotImplemented
    pieces: list = NotImplemented

    def __init__(self):
        self._reset()

    def _reset(self):
        board = Board(row_num=self.board_size, col_num=self.board_size)
        self._state = BoardState(board=board, next_piece=Piece.X)

    def reset(self):
        self._reset()
        return self.get_obs()

    def render(self, mode="human"):
        logger.info(self._state.render_str())

    def get_obs(self):
        return self._state.obs()

    @classmethod
    @lru_cache()
    def _get_next_piece(cls, next_piece):
        idx = cls.pieces.index(next_piece)
        idx = (idx + 1) % len(cls.pieces)
        return cls.pieces[idx]

    def _change_piece(self):
        self._state.next_piece = self._get_next_piece(self._state.next_piece)

    @classmethod
    @abstractmethod
    def _get_valid_actions_by_state(cls, state: BoardState) -> List[Action]:
        raise NotImplementedError

    @classmethod
    def get_valid_actions_by_obs(cls, obs) -> List[Action]:
        state = BoardState.from_obs(obs)
        return cls._get_valid_actions_by_state(state=state)

    def get_valid_actions(self) -> List[Action]:
        return self._get_valid_actions_by_state(state=self._state)

    @classmethod
    @abstractmethod
    def _on_valid_action(cls, state: BoardState, action: Action) -> TransferInfo:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _on_invalid_action(cls, state: BoardState, action: Action) -> TransferInfo:
        raise NotImplementedError

    @classmethod
    def _transfer(cls, state: BoardState, action: Action) -> TransferInfo:
        # logger.info(state)
        # breakpoint()
        valid_actions = cls._get_valid_actions_by_state(state=state)
        # logger.info(f"{valid_actions=}")
        if action not in valid_actions:
            transfer_info = cls._on_invalid_action(state, action)
        else:
            transfer_info = cls._on_valid_action(state, action)
        return transfer_info

    @classmethod
    def transfer(cls, obs: Tuple, action: Action) -> TransferInfo:
        state = BoardState.from_obs(obs)
        return cls._transfer(state=state, action=action)

    def step(self, action: Action) -> TransferInfo:
        return self._transfer(state=self._state, action=action)

    def close(self):
        pass

    def seed(self, seed):
        pass


class BaseProcess(Process):
    def __init__(self, context: Context, run_kwargs=dict(), *args, **kwargs):
        super(BaseProcess, self).__init__(*args, **kwargs)
        self.context = context
        self.run_kwargs = run_kwargs
