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
import logging
import os
from abc import abstractmethod, ABC
from enum import Enum
from typing import Tuple, List

from pydantic import BaseModel, Field

from rlb.utils import sample_by_probs

logger = logging.getLogger(__name__)


class Action(ABC, BaseModel):
    @abstractmethod
    def to_idx(self) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_idx(cls, idx):
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


class Env(ABC):
    action_num: int = NotImplemented
    action_cls: type = NotImplemented

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
    def __init__(self, name, action_num):
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
        logger.debug(f"choose action:{action_idx} with prob:{prob:2.3f}")
        action_info = ActionInfo(action_idx=action_idx, prob=prob, probs=probs)
        return action_info

    def __repr__(self):
        return self.name

    def on_episode_end(self, episode_idx: int):
        pass


class RandomAgent(Agent):
    def get_weights(self, **kwargs) -> List[float]:
        return [1 / self.action_num] * self.action_num


class Step(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    agent_name: str
    obs: Tuple
    action_idx: int
    prob: float = Field(le=1., ge=0.)
    probs: List[float]
    reward: float
    next_obs: Tuple
    is_done: bool
    extra_info: dict = {}

    @classmethod
    def from_info(cls, agent_name, action_info: ActionInfo, transfer_info: TransferInfo):
        return cls(agent_name=agent_name, **action_info.dict(), **transfer_info.dict())


class Episode(BaseModel):
    steps: List[Step] = list()
    cost: float = 0.

    @property
    def step(self):
        return len(self.steps)

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

    def ckpt_model_path(self, ckpt: int):
        return os.path.join(self.model_dir, f"model-{ckpt}.pt")


class Piece(str, Enum):
    BLANK = "_"
    X = "X"
    O = "O"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


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


class BoardEnv(Env, ABC):
    board_size: int = NotImplemented

    def __init__(self):
        self._reset()

    def _reset(self):
        self._board = Board(row_num=self.board_size, col_num=self.board_size)
        self._next_piece = Piece.X

    def reset(self):
        self._reset()
        return self.get_obs()

    def render(self, mode="human"):
        state_info = f"\n{self._board}|{self._next_piece}"
        logger.info(state_info)

    @classmethod
    def _get_obs(cls, board: Board, next_piece: Piece) -> Tuple:
        obs = []
        for row in board.rows:
            obs.append(tuple(row))
        return tuple(obs), next_piece

    def get_obs(self):
        return self._get_obs(self._board, self._next_piece)

    @classmethod
    @abstractmethod
    def _get_next_piece(cls, piece: Piece) -> Piece:
        raise NotImplementedError

    def _change_piece(self):
        self._next_piece = self._get_next_piece(self._next_piece)

    @classmethod
    def _obs2board_piece(cls, obs: Tuple) -> Tuple[Board, Piece]:
        b, next_piece = obs
        b = [list(e) for e in b]
        board = Board(row_num=cls.board_size, col_num=cls.board_size, board=b)
        return board, next_piece

    @classmethod
    @abstractmethod
    def _get_valid_actions_by_board_piece(cls, board: Board, piece: Piece) -> List[Action]:
        raise NotImplementedError

    @classmethod
    def get_valid_actions_by_obs(cls, obs) -> List:
        board, piece = cls._obs2board_piece(obs)
        return cls._get_valid_actions_by_board_piece(board, piece)

    def get_valid_actions(self) -> List[Action]:
        obs = self.get_obs()
        return self.get_valid_actions_by_obs(obs=obs)

    @classmethod
    @abstractmethod
    def _transfer_on_board_piece(cls, board: Board, piece: Piece, action: Action) -> Tuple[TransferInfo, Piece]:
        raise NotImplementedError

    @classmethod
    def transfer(cls, obs: Tuple, action: Action) -> TransferInfo:
        board, next_piece = cls._obs2board_piece(obs)
        valid_actions = cls._get_valid_actions_by_board_piece(board, next_piece)
        if action not in valid_actions:
            transfer_info, next_piece = cls._on_invalid_action(board, next_piece, action)
        else:
            transfer_info, next_piece = cls._transfer_on_board_piece(board, next_piece, action)
        return transfer_info

    @classmethod
    @abstractmethod
    def _on_invalid_action(cls, board: Board, piece: Piece,
                           action: Action) -> Tuple[TransferInfo, Piece]:
        raise NotImplementedError

    def _set_piece(self, piece: Piece):
        self._next_piece = piece

    def step(self, action: Action) -> TransferInfo:

        valid_actions = self._get_valid_actions_by_board_piece(self._board, self._next_piece)
        if action not in valid_actions:
            transfer_info, next_piece = self._on_invalid_action(self._board, self._next_piece, action)
        else:
            transfer_info, next_piece = self._transfer_on_board_piece(self._board, self._next_piece, action)
        self._set_piece(next_piece)
        return transfer_info

    def close(self):
        pass

    def seed(self, seed):
        pass
