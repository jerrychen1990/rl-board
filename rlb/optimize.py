#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     optimize.py
   Author :       chenhao
   time：          2022/3/4 16:24
   Description :
-------------------------------------------------
"""
import logging
import os
from time import sleep

from rlb.client import sample_ac_infos
from rlb.core import Context
from rlb.model import get_optimizer_cls, get_schedule_cls, ModuleActorCritic
from rlb.utils import save_torch_model

logger = logging.getLogger(__name__)


class Optimizer:
    def __init__(self, ac_model: ModuleActorCritic, context: Context, opt_kwargs, schedule_kwargs, *args, **kwargs):
        self.ac_model = ac_model
        self.context = context
        self._init_optimizer(opt_kwargs)
        self._init_schedule(schedule_kwargs)
        self.step = 0

    def _init_optimizer(self, opt_kwargs):
        optimizer_name = opt_kwargs.pop("name")
        opt_cls = get_optimizer_cls(optimizer_name=optimizer_name)
        self.optimizer = opt_cls(params=self.ac_model.parameters(), **opt_kwargs)

    def _init_schedule(self, schedule_kwargs):
        schedule_name = schedule_kwargs.pop("name")
        schedule_cls = get_schedule_cls(schedule_name=schedule_name)
        self.schedule = schedule_cls(optimizer=self.optimizer, **schedule_kwargs)

    def optimize_one_ckpt(self, step_size, sample_batch_size, actor_loss_type="ce"):
        for idx in range(step_size):
            ac_infos = sample_ac_infos(n=sample_batch_size)
            if not ac_infos:
                continue
            self.ac_model.learn_on_batch(ac_infos=ac_infos, optimizer=self.optimizer, actor_loss_type=actor_loss_type)
        self.step += step_size * sample_batch_size

    def save_model(self):
        save_torch_model(model=self.ac_model, path=self.context.ckpt_model_path(self.step))

    def optimize(self, step_size, sample_batch_size, interval=0.,
                 max_step=None, conn=None, actor_loss_type="ce"):
        self.step = 0
        while True:
            self.optimize_one_ckpt(step_size=step_size, sample_batch_size=sample_batch_size,
                                   actor_loss_typ=actor_loss_type)

            model_path = self.context.ckpt_model_path(ckpt=self.step)
            logger.info(f"saving model with ckpt:{self.step}")
            save_torch_model(model=self.ac_model, path=model_path)
            self.conn.send((self.step, model_path))
            sleep(interval)
            if max_step and self.step >= max_step:
                break
            if conn:
                conn.recv()
            self.schedule.step()
