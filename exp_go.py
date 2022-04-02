#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     exp_go.py
   Author :       chenhao
   time：          2022/3/15 17:05
   Description :
-------------------------------------------------
"""
import copy
import logging

from rlb.core import RandomAgent
from rlb.engine import run_board_games, compare_agents_with_detail
from rlb.go import Go4
from rlb.human import HumanAgent
from rlb.minmax import MinMaxAgent

logger = logging.getLogger(__name__)

env = Go4()

env_cls = env.__class__
action_num = env_cls.action_num
action_cls = env_cls.action_cls

minmax_agent = MinMaxAgent(name="minmax_agent", action_num=action_num)
# minmax_agent.train(env=env)
# human_agent_cls = get_human_agent_cls(env_cls)
human_agent = HumanAgent(name="human_agent", action_num=action_num, action_cls=action_cls)


def test_env(episodes=20):
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

def test_human_random(episodes=5):
    rand_agent = RandomAgent(name="rand_agent", action_num=action_num)

    history, scoreboard = run_board_games(agents=[human_agent, rand_agent], episodes=episodes, shuffle=True,
                                          env=env, mode="test", is_render=True)
    logger.info(scoreboard)

def test_min_max(episodes=20, value_decay=1.0):
    minmax_agent1 = MinMaxAgent(name="minmax_agent1", action_num=action_num)
    minmax_agent1.train(env_cls=env_cls, value_decay=value_decay)

    minmax_agent2 = copy.deepcopy(minmax_agent1)
    setattr(minmax_agent2, "_name", "minmax_agent2")

    scoreboard, win_rate = compare_agents_with_detail(agent=minmax_agent1, tgt_agent=minmax_agent2, env=env,
                                                      episodes=episodes)

    logger.info(f"{scoreboard=}, {win_rate=:2.3f}")


if __name__ == '__main__':
    test_human()
    # test_human_random()
    # test_env()
    # test_min_max(episodes=20, value_decay=1.0)
