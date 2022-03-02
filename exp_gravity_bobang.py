#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     exp_gravity_gobang.py
   Author :       chenhao
   time：          2022/2/15 11:12
   Description :
-------------------------------------------------
"""
import logging
from pathlib import Path

from snippets import jload, jdumps

from rlb.core import RandomAgent
from rlb.deep_rl import ReinforcementLearning
from rlb.engine import run_episodes
from rlb.gravity_gobang import GravityGoBang3
from rlb.human import GravityGoBangHumanAgent
from rlb.minmax import MinMaxAgent

logger = logging.getLogger(__name__)

env = GravityGoBang3()
action_num = env.action_num

minmax_agent = MinMaxAgent(name="minmax_agent", action_num=action_num)
# minmax_agent.train(env=env)
human_agent = GravityGoBangHumanAgent(name="human_agent", action_num=action_num)


def test_env():
    rand1 = RandomAgent(name="rand1", action_num=action_num)
    rand2 = RandomAgent(name="rand2", action_num=action_num)

    episodes = 1

    history, scoreboard = run_episodes(agents=[rand1, rand2], episodes=episodes, env=env,
                                       mode="test", is_render=True, max_step=20)
    logger.info(scoreboard)


def test_human():

    episodes = 5

    history, scoreboard = run_episodes(agents=[human_agent, human_agent], episodes=episodes,
                                       env=env, mode="test", is_render=True)
    logger.info(scoreboard)


def test_min_max():
    episodes = 20

    history, scoreboard = run_episodes(agents=[minmax_agent, human_agent], episodes=episodes,
                                       env=env, mode="test", is_render=True)

    # against_agent = MinMaxAgent(name="against_agent", action_num=action_num)
    # against_agent.train(env=env)
    #
    # history, scoreboard = run_episodes(agents=[against_agent, minmax_agent], episodes=episodes,
    #                                    env=env, mode="test", is_render=False, shuffle=False)

    logger.info(scoreboard)


def test_deep_rl():
    config_path = "configs/ggobang/supervised_rl.json"
    # config_path = "configs/ggobang/supervised_rl_simple.json"
    config_path = "configs/ggobang/rl.json"



    exp_name = Path(config_path).stem

    config = jload(config_path)
    logger.info(jdumps(config))
    experiment = ReinforcementLearning(name=exp_name, config=config, env=env)
    experiment.run()
    # model_path = f"experiments/TicTacToe/{exp_name}/models/best_model.pt"
    # eval_model_mcst(model_path)


if __name__ == "__main__":
    # test_env()
    # test_human()
    # test_min_max()
    test_deep_rl()
