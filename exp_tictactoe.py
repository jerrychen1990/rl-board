#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     exp_tictactoe.py
   Author :       chenhao
   time：          2022/2/15 11:12
   Description :
-------------------------------------------------
"""
import logging
from pathlib import Path

import torch
from snippets import jload, jdumps

from rlb.core import RandomAgent
from rlb.deep_rl import ReinforcementLearning
from rlb.engine import run_episodes
from rlb.human import TicTacToeHumanAgent
from rlb.mcst import MCSTAgent
from rlb.minmax import MinMaxAgent
from rlb.tictactoe import TicTacToe

logger = logging.getLogger(__name__)

env = TicTacToe()
action_num = env.action_num
minmax_agent = MinMaxAgent(name="minmax_agent", action_num=action_num)
minmax_agent.train(env=env)
human_agent = TicTacToeHumanAgent(name="human_agent", action_num=action_num)


def eval_agent(agent, episodes=20):
    logger.info(f"evaluating agent:{agent} with {episodes} episodes")
    history, scoreboard = run_episodes(agents=[agent, minmax_agent], episodes=episodes, env=env, mode="test",
                                       is_render=False)
    logger.info(scoreboard)
    # agent.clear_mcst()


def debug_agent(agent, episodes=5):
    logging.getLogger("rlb.mcst").setLevel(logging.DEBUG)
    logger.info(f"debuging agent:{agent} with {episodes} episodes")
    history, scoreboard = run_episodes(agents=[agent, human_agent], episodes=episodes, env=env, mode="test",
                                       is_render=True)
    # agent.clear_mcst()
    logger.info(scoreboard)
    logging.getLogger("rlb.mcst").setLevel(logging.INFO)


def test_env():
    rand1 = RandomAgent(name="rand1", action_num=action_num)
    rand2 = RandomAgent(name="rand2", action_num=action_num)

    episodes = 10

    history, scoreboard = run_episodes(agents=[rand1, rand2], episodes=episodes, env=env, mode="test", is_render=True)
    logger.info(scoreboard)


def test_minmax():
    history, scoreboard = run_episodes(agents=[minmax_agent, human_agent], episodes=5, env=env, mode="test",
                                       is_render=True)
    logger.info(scoreboard)


def test_deep_rl():
    config_path = "configs/tictactoe/alpha_tictactoe.json"
    # config_path = "configs/alpha_tictactoe_simple.json"

    # config_path = "configs/supervised_tictactoe.json"

    exp_name = Path(config_path).stem

    config = jload(config_path)
    logger.info(jdumps(config))
    env = TicTacToe()
    experiment = ReinforcementLearning(name=exp_name, config=config, env=env)
    experiment.run()
    model_path = f"experiments/TicTacToe/{exp_name}/models/best_model.pt"
    eval_model_mcst(model_path)

def eval_model_mcst(model_path):
    model = torch.load(model_path)
    agent = MCSTAgent(name="mcst_agent", env_cls=TicTacToe, ac_model=model, simulate_num=50)
    eval_agent(agent, 100)
    debug_agent(agent, 5)




if __name__ == "__main__":
    # test_env()
    # test_ac()
    # test_minmax()
    test_deep_rl()
    # eval_model_mcst()
