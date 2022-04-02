#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     self_play.py
   Author :       chenhao
   time：          2022/3/4 15:53
   Description :
-------------------------------------------------
"""
import logging
import os

import torch

from rlb.board_core import BoardEnv, Episode, Env
from rlb.client import get_ckpt, puts_ac_infos
from rlb.core import Agent, Context
from rlb.engine import AbsCallback, RecordCallback, run_board_games

logger = logging.getLogger(__name__)
cur_dir = os.path.abspath(os.path.dirname(__file__))


class SelfPlayCallback(AbsCallback):
    def __init__(self, agent, value_decay):
        self.current_ckpt = 0
        self.agent = agent
        self.value_decay = value_decay

    def _update_buffer(self, episode: Episode):
        ac_infos = episode.to_ac_infos(self.value_decay)
        puts_ac_infos(ac_infos)

    def _update_agent_model(self):
        ckpt_info = get_ckpt()
        if ckpt_info.get("ckpt", 0) > self.current_ckpt:
            model_path = ckpt_info["model_path"]
            logger.info(f"replace ac_model with ckpt_info:{ckpt_info}")
            ac_model = torch.load(model_path)
            self.agent.update_model(ac_model=ac_model)
            self.current_ckpt = ckpt_info["ckpt"]

    def on_episode_end(self, episode_idx, episode: Episode):
        self._update_buffer(episode)
        self._update_agent_model()


def self_play(agent: Agent, env: BoardEnv, episodes: int, context: Context,
              value_decay=1., record_episode_size=None, show_episode_size=None, is_render=False):
    agents = [agent, agent]

    callbacks = [SelfPlayCallback(agent=agent, value_decay=value_decay)]
    if record_episode_size:
        record_callback = RecordCallback(record_dir=context.record_dir, episode_step_size=record_episode_size)
        callbacks.append(record_callback)

    history, scoreboard = run_board_games(agents=agents, episodes=episodes, env=env, mode="train",
                                          shuffle=True, is_render=is_render, callbacks=callbacks,
                                          show_episode_size=show_episode_size)
    return history, scoreboard


class SelfPlayer:
    def __init__(self, agent: Agent, env: Env, context: Context):
        self.agent = agent
        self.env = env
        self.context = context

    def self_play(self, episodes: int, value_decay=1.,
                  record_episode_size=None, show_episode_size=None, is_render=False):
        return self_play(context=self.context, agent=self.agent, env=self.env, episodes=episodes,
                         value_decay=value_decay, record_episode_size=record_episode_size,
                         show_episode_size=show_episode_size,
                         is_render=is_render)

    def run(self):
        return self_play(context=self.context, agent=self.agent, env=self.env, **self.run_kwargs)
