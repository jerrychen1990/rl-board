#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_env.py
   Author :       chenhao
   time：          2022/4/7 16:26
   Description :
-------------------------------------------------
"""
import copy
import os
from pathlib import Path

from rlb.core import *
from rlb.drl import ReinforcementLearning
from rlb.engine import run_board_games, compare_agents_with_detail
from rlb.gobang import TICTACTOE
from rlb.othello import OTHELLO4
from snippets import LoggingLevelContext, jload, jdumps

from rlb.human import HumanAgent
from rlb.minmax import MinMaxAgent

logger = logging.getLogger(__name__)
# env = TICTACTOE
env = OTHELLO4
action_num = env.action_num
board_size = env.board_size


def test_env(episodes=2, level=logging.INFO):
    with LoggingLevelContext(level=level, loggers=[logging.root]):
        rand1 = RandomAgent(name="rand1", action_num=action_num)
        rand2 = RandomAgent(name="rand2", action_num=action_num)
        history, scoreboard = run_board_games(agents=[rand1, rand2], episodes=episodes, env=env,
                                              mode="test", is_render=True, max_step=20)
        logger.info(scoreboard)


def test_human(episodes=5):
    human_agent1 = HumanAgent(name="human_agent1", action_num=action_num, board_size=board_size)
    human_agent2 = HumanAgent(name="human_agent2", action_num=action_num, board_size=board_size)

    history, scoreboard = run_board_games(agents=[human_agent1, human_agent2], episodes=episodes,
                                          env=env, mode="test", is_render=True)
    logger.info(scoreboard)


def test_min_max(episodes=20, value_decay=1.0, overwrite_cache=False):
    minmax_agent1 = MinMaxAgent(name="minmax_agent1", action_num=action_num)
    minmax_agent1.train(env=env, value_decay=value_decay, overwrite_cache=overwrite_cache)
    ac_infos = minmax_agent1.gen_ac_infos()
    logger.info(f"generate {len(ac_infos)} ac_infos")
    minmax_agent2 = copy.copy(minmax_agent1)
    minmax_agent2.name = "minmax_agent2"

    logger.info("compare minmax agents")
    scoreboard, win_rate = compare_agents_with_detail(agent=minmax_agent1, tgt_agent=minmax_agent2, env=env,
                                                      episodes=episodes)

    logger.info(f"{scoreboard=}, {win_rate=:2.3f}")

    logger.info("compare minmax agent with human")
    human_agent = HumanAgent(name="human_agent", action_num=action_num, board_size=board_size)
    history, scoreboard = run_board_games(agents=[minmax_agent1, human_agent], episodes=episodes,
                                          env=env, mode="test", is_render=True, is_shuffle=True)
    logger.info(scoreboard)


def test_drl(config_name, eval_episodes=100):
    config_path = os.path.join("configs", env.name, config_name)
    logger.info(f"{config_path=}")

    exp_name = Path(config_path).stem

    config = jload(config_path)
    logger.info(jdumps(config))
    experiment = ReinforcementLearning(name=exp_name, config=config, env=env, overwrite_time=True)
    experiment.train()
    experiment.eval_with_perfect_agent(episodes=eval_episodes)
    experiment.eval_with_random_agent(episodes=eval_episodes)


if __name__ == "__main__":
    # test_env(episodes=1)
    # test_human(episodes=5)
    test_min_max(episodes=20, value_decay=1., overwrite_cache=False)

    # config_name = "rl_sync_az.json"
    # test_drl(config_name=config_name, eval_episodes=100)
