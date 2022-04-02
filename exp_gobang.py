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
import os
from pathlib import Path

from snippets import jload, jdumps, LoggingLevelContext
from torch.optim import Adam

from rlb.core import RandomAgent
from rlb.drl import ReinforcementLearning
from rlb.engine import run_board_games, compare_agents_with_detail
from rlb.gobang import *
from rlb.human import HumanAgent
from rlb.minmax import MinMaxAgent
from rlb.model import build_ac_model

logger = logging.getLogger(__name__)

# env = GravityGoBang33()
# env = GravityGoBang34()
env = TicTacToe()

env_cls = env.__class__
action_num = env_cls.action_num
action_cls = env_cls.action_cls

minmax_agent = MinMaxAgent(name="minmax_agent", action_num=action_num)
# minmax_agent.train(env=env)
# human_agent_cls = get_human_agent_cls(env_cls)
human_agent = HumanAgent(name="human_agent", action_num=action_num, action_cls=action_cls)


def test_env(episodes=2, level=logging.INFO):
    with LoggingLevelContext(level=level, loggers=[logging.root, logging.getLogger("rlb.bord_core"),
                                                   logging.getLogger("rlb.engine")]):
        rand1 = RandomAgent(name="rand1", action_num=action_num)
        rand2 = RandomAgent(name="rand2", action_num=action_num)
        history, scoreboard = run_board_games(agents=[rand1, rand2], episodes=episodes, env=env,
                                              mode="test", is_render=True, max_step=20)
        logger.info(scoreboard)


def test_human(episodes=5):
    human_agent1 = HumanAgent(name="human_agent1", action_num=action_num, action_cls=action_cls)

    history, scoreboard = run_board_games(agents=[human_agent, human_agent1], episodes=episodes,
                                          env=env, mode="test", is_render=True)
    logger.info(scoreboard)


def test_min_max(episodes=20, value_decay=1.0, overwrite_cache=False):
    minmax_agent1 = MinMaxAgent(name="minmax_agent1", action_num=action_num)
    minmax_agent1.train(env_cls=env_cls, value_decay=value_decay, overwrite_cache=overwrite_cache)
    ac_infos = minmax_agent1.gen_ac_infos()
    key_ac_infos = [e for e in ac_infos if e.is_in_path]
    logger.info(f"generate {len(ac_infos)} ac_infos, {len(key_ac_infos)} key ac_infos")

    minmax_agent2 = copy.deepcopy(minmax_agent1)
    setattr(minmax_agent2, "_name", "minmax_agent2")

    scoreboard, win_rate = compare_agents_with_detail(agent=minmax_agent1, tgt_agent=minmax_agent2, env=env,
                                                      episodes=episodes)

    logger.info(f"{scoreboard=}, {win_rate=:2.3f}")


def test_min_max_with_human(episodes=5, value_decay=1., overwrite_cache=False):
    minmax_agent1 = MinMaxAgent(name="minmax_agent1", action_num=action_num)
    minmax_agent1.train(env_cls=env_cls, value_decay=value_decay, overwrite_cache=overwrite_cache)

    history, scoreboard = run_board_games(agents=[minmax_agent1, human_agent], episodes=episodes,
                                          env=env, mode="test", is_render=True, shuffle=True)
    logger.info(scoreboard)


def train_ac():
    minmax_agent.train(env_cls=env_cls, value_decay=0.9)
    ac_infos = minmax_agent.gen_ac_infos()

    config_name = "rl_sync_az_supervise.json"
    config_path = os.path.join("configs", env.__class__.__name__, config_name)
    logger.info(config_path)
    config = jload(config_path)
    ac_model = build_ac_model(env_cls=env_cls, **config["ac_model_kwargs"])
    optimizer = Adam(params=ac_model.parameters(), lr=1e-2)

    ac_model.train(ac_infos=ac_infos, optimizer=optimizer, epochs=100, show_lines=20)


def test_drl():
    # config_name = "rl_sync.json"
    config_name = "rl_sync_az.json"
    # config_name = "rl_sync_az_supervise.json"

    config_path = os.path.join("configs", env.__class__.__name__, config_name)
    logger.info(f"{config_path=}")

    exp_name = Path(config_path).stem

    config = jload(config_path)
    logger.info(jdumps(config))
    experiment = ReinforcementLearning(name=exp_name, config=config, env=env, overwrite_time=True)
    experiment.train()

    eval_episodes = 100
    experiment.eval_with_perfect_agent(episodes=eval_episodes)
    experiment.eval_with_random_agent(episodes=eval_episodes)

    # model_path = f"experiments/TicTacToe/{exp_name}/models/best_model.pt"
    # eval_model_mcst(model_path)


if __name__ == "__main__":
    # test_env(episodes=10, level=logging.INFO)
    # test_human(episodes=5)
    # test_min_max(episodes=100, value_decay=.9, overwrite_cache=True)

    # test_min_max_with_human(episodes=5, value_decay=0.9)
    test_drl()
    # train_ac()
