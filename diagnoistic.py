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
from snippets import jload_lines, jload, LoggingLevelContext, flat

from rlb.core import Context, Episode, State
from rlb.engine import run_board_games
from rlb.evaluate import evaluate_episodes
from rlb.gobang import TICTACTOE
from rlb.mcst import MCSTAgent
from rlb.minmax import MinMaxAgent
from rlb.model import load_ac_model, build_ac_model
from rlb.utils import format_dict

logger = logging.getLogger(__name__)


env = TICTACTOE
action_num = env.action_num

minmax_agent = MinMaxAgent(name="minmax_agent", action_num=action_num)
# minmax_agent.train(env=env, value_decay=0.95)
# perfect_ac_infos = minmax_agent.gen_ac_infos()

config_name = "rl_sync_az"

base_dir = os.path.join("experiments/", env.name, config_name)
context = Context(base_dir=base_dir)
config = jload(context.config_path)


def explore_self_play(ckpt, episodes, level=logging.DEBUG, is_render=True):
    ac_model_kwargs = config["ac_model_kwargs"]
    if ckpt:
        ac_model = load_ac_model(context=context, ckpt=-1)
    else:
        ac_model = build_ac_model(env=env, **ac_model_kwargs)

    mcst_agent = MCSTAgent(name="mcst_agent", env=env,
                           ac_model=ac_model, **config["mcst_kwargs"])
    agents = [mcst_agent, mcst_agent]

    with LoggingLevelContext(level=level, loggers=[logging.root, logging.getLogger("rlb.mcst")]):
        history, scoreboard = run_board_games(agents=agents, episodes=episodes, env=env, mode="train",
                                              is_shuffle=True, is_render=is_render, callbacks=[])
    return history, scoreboard


def explore_records():
    eval_rs_list = []
    minmax_agent.train(env=env, value_decay=0.95, overwrite_cache=False)
    perfect_ac_infos = minmax_agent.gen_ac_infos()
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

#
# def get_episodes_statistic(episodes: List[Episode], top=10):
#     info_dict = dict()
#     for step in flat([e.steps for e in episodes]):
#         state = step.state
#         if state not in info_dict:
#             info_dict[state] = collections.defaultdict(list)
#         action_idx = step.action_idx
#         info_dict[state][action_idx].append(step.probs)
#     info_list = sorted(info_dict.items(), key=lambda x: sum([len(e) for e in x[1].values()]), reverse=True)
#     for state, action_info in info_list[:top]:
#         all_probs = list(flat(action_info.values()))
#         avg_probs = np.array(all_probs).mean(axis=0).tolist()
#         logger.info(f"state:{state}, total_visit:{len(all_probs)}")
#         prob_info = ", ".join(f"{action_cls.from_idx(k)}={v:2.3f}" for k, v in
#                               sorted(enumerate(avg_probs), key=lambda x: x[1], reverse=True))
#         logger.info(f"avg_probs:[{prob_info}]")
#
#         for action_idx, probs in sorted(action_info.items(), key=lambda x: len(x[1]), reverse=True):
#             # avg_probs = np.array(probs).mean(axis=0).tolist()
#             # avg_prob_info = dict(enumerate(avg_probs))
#             logger.info(f"action:{action_cls.from_idx(action_idx)}, {len(probs)} times")
#         logger.info("*" * 50)


# def explore_episode(episode: Episode):
#     for step in episode.steps:
#         state = step.state
#         logger.info(state)
#         logger.info(f"action:{action_cls.from_idx(step.action_idx)}")
#         prob_info = ", ".join(f"{action_cls.from_idx(k)}={v:2.3f}" for k, v in
#                               sorted(enumerate(step.probs), key=lambda x: x[1], reverse=True))
#         logger.info(f"probs:[{prob_info}]")
#         logger.info(f"{step.is_done=}, {step.extra_info=}")
#         logger.info("*"*50)
#     logger.info(step.next_state)


# def explore_minmax():
#     minmax_agent.train(env=env, value_decay=1., overwrite_cache=True)
#     node = minmax_agent.root
#     while True:
#         logger.info(f"{State.from_obs(node.state)}, value={node.value}")
#         for action, n in node.children.items():
#             logger.info(f"action:{action}, [{action in node.max_actions}], {n.value}")
#
#         action_idx = input()
#         action = action_cls.from_idx(action_idx)
#         node = node.children[action]
#
#         logger.info("*" * 50)


if __name__ == "__main__":
    # explore steps
    # explore_records()
    # explore_minmax()

    # explore_episodes detail
    # name = "400-600.jsonl"
    # path = os.path.join(context.record_dir, name)
    # episodes = jload_lines(path)
    # episodes = [Episode(**e) for e in episodes]
    # get_episodes_statistic(episodes=episodes, top=10)

    # explore self play
    history, scoreboard = explore_self_play(ckpt=False, episodes=20, level=logging.INFO, is_render=False)
