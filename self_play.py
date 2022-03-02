#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     self_play.py
   Author :       chenhao
   time：          2022/3/2 16:51
   Description :
-------------------------------------------------
"""

import logging

import torch

from rlb.client import reset_buffer
from rlb.core import Context, BoardEnv
from rlb.deep_rl import SelfPlayProcess, new_ac_model
from rlb.gravity_gobang import GravityGoBang3
from rlb.mcst import MCSTAgent

logger = logging.getLogger(__name__)


def run_self_play(env: BoardEnv, base_dir: str, episodes, ckpt=None, model_kwargs=dict(), mcst_kwargs=dict()):
    if base_dir:
        context = Context(base_dir=base_dir)
        if ckpt:
            ckpt_path = context.ckpt_model_path(ckpt=ckpt)
        else:
            ckpt_path = context.best_model_path
    else:
        ckpt_path = None
    ckpt_path = None

    ac_model = new_ac_model(env=env, ckpt_path=ckpt_path, **model_kwargs)

    mcst_agent = MCSTAgent(name="self_play_mcst_agent", env_cls=env.__class__,
                           ac_model=ac_model, **mcst_kwargs)
    logger.info("start self play")
    logging.getLogger("rlb.mcst").setLevel(logging.DEBUG)

    self_play_process = SelfPlayProcess(name="self_play_process", agent=mcst_agent,
                                        env=env, context=context, run_kwargs={})
    self_play_process.self_play(episodes=episodes)

    logging.getLogger("rlb.mcst").setLevel(logging.INFO)

    logger.info("self play ends")


if __name__ == "__main__":
    logger
    env = GravityGoBang3()
    base_dir = "./experiments/GravityGoBang3/rl"
    episodes = 2
    ckpt = 100
    mcst_kwargs = {
        "c": 2,
        "tau_kwargs": {
            "tau": 1,
            "schedule": [
                [
                    5,
                    3
                ]
            ]
        },
        "simulate_num": 15,
        "noise_kwargs": {
            "noise_rate": 0.25,
            "dirichlet_alpha": 0.1
        }
    }

    ac_model_kwargs = {
        "torch_kwargs": {
            "dims": [
                64,
                128,
                64,
                64,
                32
            ]
        },
        "model_type": "MLP"
    }
    run_self_play(env=env, base_dir=base_dir, episodes=episodes, ckpt=ckpt,
                  mcst_kwargs=mcst_kwargs, model_kwargs=ac_model_kwargs)
