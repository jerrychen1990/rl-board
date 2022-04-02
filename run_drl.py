#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_drl.py
   Author :       chenhao
   time：          2022/3/15 14:59
   Description :
-------------------------------------------------
"""
import os
import logging
from pathlib import Path

import click
from snippets import jload, jdumps

from rlb.drl import ReinforcementLearning
from rlb.gobang import __all__env_cls__

env_cls_dict = {e.__name__: e for e in __all__env_cls__}

logger = logging.getLogger(__name__)


@click.command()
@click.option('--env_cls_name', help='env_cls')
@click.option('--config_name', help='config_name')
def test_drl(env_cls_name, config_name, eval_episodes=100):
    env_cls = env_cls_dict[env_cls_name]
    env = env_cls()
    config_path = os.path.join("configs", env.__class__.__name__, config_name)
    logger.info(f"{config_path=}")

    exp_name = Path(config_path).stem

    config = jload(config_path)
    logger.info("experiment config")
    logger.info(jdumps(config))
    experiment = ReinforcementLearning(name=exp_name, config=config, env=env, overwrite_time=True)
    experiment.train()
    experiment.eval_with_perfect_agent(episodes=eval_episodes)
    experiment.eval_with_random_agent(episodes=eval_episodes)


if __name__ == "__main__":
    test_drl()
