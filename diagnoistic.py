#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     diagnostic.py
   Author :       chenhao
   time：          2022/3/14 15:09
   Description :
-------------------------------------------------
"""
import collections
import logging
import os
from typing import List

import numpy as np
from snippets import flat, jload_lines, jload, LoggingLevelContext
from tqdm import tqdm

from rlb.core import Episode, BoardState, Context, ACInfo
from rlb.evaluate import eval_probs, evaluate_episodes
from rlb.gobang import GravityGoBang33
from rlb.mcst import MCSTAgent
from rlb.minmax import MinMaxAgent
from rlb.model import load_ac_model, build_ac_model
from rlb.self_play import SelfPlayer
from rlb.utils import entropy, format_dict

logger = logging.getLogger(__name__)

env = GravityGoBang33()
# env = TicTacToe()

env_cls = env.__class__
action_num = env.action_num
action_cls = env_cls.action_cls

minmax_agent = MinMaxAgent(name="minmax_agent", action_num=action_num)
minmax_agent.train(env_cls=env_cls, value_decay=0.95)
perfect_ac_infos = minmax_agent.gen_ac_infos()

config_name = "rl_sync_az"

base_dir = os.path.join("experiments/", env_cls.__name__, config_name)
context = Context(base_dir=base_dir)
config = jload(context.config_path)


def explore_self_play(ckpt, episodes, value_decay):
    ac_model_kwargs = config["ac_model_kwargs"]
    if ckpt:
        ac_model = load_ac_model(context=context, ckpt=-1)
    else:
        ac_model = build_ac_model(env_cls=env_cls, **ac_model_kwargs)

    mcst_agent = MCSTAgent(name="mcst_agent", env_cls=env_cls,
                           ac_model=ac_model, **config["mcst_kwargs"])
    self_player = SelfPlayer(agent=mcst_agent, env=env, context=context)

    with LoggingLevelContext(level=logging.DEBUG, loggers=[logging.root, logging.getLogger("rlb.mcst")]):
        history, scoreboard = self_player.self_play(episodes=episodes, value_decay=value_decay,
                                                    show_episode_size=max(episodes // 5, 1))
    return history, scoreboard


def explore_records():
    eval_rs_list = []
    for file in sorted(os.listdir(context.record_dir), key=lambda x: eval(x.split("-")[0])):
        path = os.path.join(context.record_dir, file)
        logger.info(path)
        episodes = jload_lines(path)
        episodes = [Episode(**e) for e in episodes]
        eval_rs = evaluate_episodes(episodes=episodes, perfect_ac_infos=perfect_ac_infos)
        #     logger.info(format_dict(eval_rs))
        eval_rs_list.append((file, eval_rs))
    for name, eval_rs in eval_rs_list:
        logger.info(f"{name}: {format_dict(eval_rs)}")


if __name__ == "__main__":
    # explore steps

    explore_records()

    # explore self play
    # explore_self_play(ckpt=True, episodes=5, value_decay=0.95)
