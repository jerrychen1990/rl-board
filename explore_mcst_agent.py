#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     explore_mcst_agent.py
   Author :       chenhao
   time：          2022/3/1 17:07
   Description :
-------------------------------------------------
"""
import os
import logging

import torch

from rlb.core import Env
from rlb.engine import run_episodes
from rlb.gravity_gobang import GravityGoBang3
from rlb.human import GravityGoBangHumanAgent
from rlb.mcst import MCSTAgent

logger = logging.getLogger(__name__)

env = GravityGoBang3()
action_num = env.action_num
human_agent = GravityGoBangHumanAgent(name="human_agent", action_num=action_num)


def explore_mcst_agent(experiment_name: str, env: Env, episodes, mcst_kwargs: dict, ckpt=None):
    if ckpt:
        model_path = os.path.join("experiments", env.__class__.__name__, experiment_name, f"models/model-{ckpt}.pt")
    else:
        model_path = os.path.join("experiments", env.__class__.__name__, experiment_name, "models/best_model.pt")
    logger.info(f"loading model from {model_path}")
    ac_model = torch.load(model_path)
    mcst_agent = MCSTAgent(name="mcst_agent", env_cls=env.__class__, ac_model=ac_model, **mcst_kwargs)

    logging.getLogger("rlb.mcst").setLevel(logging.DEBUG)
    logger.info(f"debuging agent:{mcst_agent} with {episodes} episodes")
    history, scoreboard = run_episodes(agents=[mcst_agent, human_agent], episodes=episodes, env=env, mode="test",
                                       is_render=True)
    # agent.clear_mcst()
    logger.info(scoreboard)
    logging.getLogger("rlb.mcst").setLevel(logging.INFO)


def explore_self_play(experiment_name: str, env: Env, episodes, mcst_kwargs: dict, tau_kwargs: dict, ckpt=None):
    if ckpt:
        model_path = os.path.join("experiments", env.__class__.__name__, experiment_name, f"models/model-{ckpt}.pt")
    else:
        model_path = os.path.join("experiments", env.__class__.__name__, experiment_name, "models/best_model.pt")
    logger.info(f"loading model from {model_path}")
    ac_model = torch.load(model_path)
    mcst_agent = MCSTAgent(name="mcst_agent", env_cls=env.__class__, tau_kwargs=tau_kwargs, ac_model=ac_model,
                           **mcst_kwargs)

    logging.getLogger("rlb.mcst").setLevel(logging.DEBUG)
    logger.info(f"debuging agent:{mcst_agent} with {episodes} episodes")
    history, scoreboard = run_episodes(agents=[mcst_agent, mcst_agent], episodes=episodes, env=env, mode="train",
                                       is_render=True, shuffle=False)
    # agent.clear_mcst()
    logger.info(scoreboard)
    logging.getLogger("rlb.mcst").setLevel(logging.INFO)


if __name__ == "__main__":
    # explore_mcst_agent(experiment_name="rl", env=env, episodes=5, mcst_kwargs=dict(simulate_num=20), ckpt=100)
    explore_self_play(experiment_name="rl", env=env, episodes=2, mcst_kwargs=dict(simulate_num=10),
                      tau_kwargs=dict(tau=1., schedule=[(4, 3)]), ckpt=100)
