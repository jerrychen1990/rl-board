#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     deep_rl.py
   Author :       chenhao
   time：          2022/2/23 17:15
   Description :
-------------------------------------------------
"""
import copy
import logging
import os
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from time import sleep
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from snippets import jdump, get_batched_data, ensure_file_path, get_current_datetime_str
from torch.nn import Module
from tqdm import tqdm

from rlb.client import puts_steps, get_ckpt, reset_buffer, sample_steps, add_ckpt
from rlb.constant import DRAW
from rlb.core import Episode, Context, Step, Agent, BoardEnv
from rlb.engine import AbsCallback, run_episodes
from rlb.mcst import MCSTAgent
from rlb.minmax import MinMaxAgent
from rlb.model import get_model_cls, eval_ac_model, get_optimizer_cls, get_schedule_cls

logger = logging.getLogger(__name__)
cur_dir = os.path.abspath(os.path.dirname(__file__))


class BaseProcess(Process):
    def __init__(self, context: Context, run_kwargs, *args, **kwargs):
        super(BaseProcess, self).__init__(*args, **kwargs)
        self.context = context
        self.run_kwargs = run_kwargs


class SelfPlayProcess(BaseProcess):
    def __init__(self, agent, env, *args, **kwargs):
        super(SelfPlayProcess, self).__init__(*args, **kwargs)
        self.agent = agent
        self.env = env

    class SelfPlayCallback(AbsCallback):
        def __init__(self, agent):
            self.current_ckpt = 0
            self.agent = agent

        @staticmethod
        def _update_buffer(episode):
            steps = episode.steps
            last_step = steps[-1]
            value = 1 if last_step.extra_info.get("is_win") else 0
            for step in steps[::-1]:
                step.extra_info.update(value=value)
                value = - value
            puts_steps(steps)

        def _update_agent_model(self):
            ckpt_info = get_ckpt()
            if ckpt_info.get("ckpt", 0) > self.current_ckpt:
                model_path = ckpt_info["model_path"]
                logger.info(f"replace ac_model with ckpt_info:{ckpt_info}")
                ac_model = torch.load(model_path)
                self.agent.update_model(ac_model=ac_model)
                self.current_ckpt = ckpt_info["ckpt"]

        def on_episode_end(self, episode_idx, episode: Episode):
            self._update_buffer(episode)
            self._update_agent_model()

    def self_play(self, episodes: int):
        agents = [self.agent, self.agent]

        callbacks = [self.SelfPlayCallback(agent=self.agent)]

        history, scoreboard = run_episodes(agents=agents, episodes=episodes, env=self.env, mode="train",
                                           shuffle=False, is_render=False, callbacks=callbacks,
                                           show_episode_size=episodes // 20)
        return history

    def run(self):
        return self.self_play(**self.run_kwargs)


def eval_model_on_steps(ac_model: Module, steps: List[Step]):
    obs = torch.stack([ac_model.obs2tensor(s.obs) for s in steps])
    tgt_probs = torch.from_numpy(np.array([s.probs for s in steps])).float()
    tgt_values = torch.from_numpy(np.array([s.extra_info["value"] for s in steps])).float()

    weights, values = ac_model.forward(obs)
    probs = F.softmax(weights, dim=-1).detach()
    actor_loss = F.cross_entropy(weights, tgt_probs)
    critic_loss = F.mse_loss(values, tgt_values)
    hard_probs = torch.nn.functional.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.shape[-1]).float()

    loss = actor_loss + critic_loss

    acc = ((tgt_probs > 0).int() * probs).sum(axis=-1).mean()
    hard_acc = ((tgt_probs > 0).int() * hard_probs).sum(axis=-1).mean()

    return acc, hard_acc, loss, actor_loss, critic_loss


def compare_agents(agent: Agent, tgt_agent: Agent, env, episodes):
    history, score_board = run_episodes(agents=[agent, tgt_agent], env=env, episodes=episodes, mode="test")
    # logger.info(score_board)
    draw_score = score_board.get(DRAW, 0)
    score = score_board.get(agent, 0)
    tgt_score = score_board.get(tgt_agent, 0)
    win_rate = (score + draw_score * 0.5) / (score + draw_score + tgt_score)
    logger.info(f"score_board:{score_board}, win_rate:{win_rate:2.3f}")
    return win_rate


class OptimizeProcess(BaseProcess):
    def __init__(self, ac_model: Module, opt_kwargs, schedule_kwargs, conn, *args, **kwargs):
        super(OptimizeProcess, self).__init__(*args, **kwargs)
        self.ac_model = ac_model
        self._init_optimizer(opt_kwargs)
        self._init_schedule(schedule_kwargs)
        self.conn = conn

    def _init_optimizer(self, opt_kwargs):
        optimizer_name = opt_kwargs.pop("name")
        opt_cls = get_optimizer_cls(optimizer_name=optimizer_name)
        self.optimizer = opt_cls(params=self.ac_model.parameters(), **opt_kwargs)

    def _init_schedule(self, schedule_kwargs):
        schedule_name = schedule_kwargs.pop("name")
        schedule_cls = get_schedule_cls(schedule_name=schedule_name)
        self.schedule = schedule_cls(optimizer=self.optimizer, **schedule_kwargs)

    def learn_on_steps(self, steps: List[Step], mini_batch_size):
        if not mini_batch_size:
            mini_batch_size = len(steps)

        actor_losses, critic_losses, losses, accs, hard_accs = [], [], [], [], []
        for batch in get_batched_data(steps, mini_batch_size):
            acc, hard_acc, loss, actor_loss, critic_loss = eval_model_on_steps(ac_model=self.ac_model, steps=batch)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.ac_model.parameters():  # clip防止梯度爆炸
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            actor_losses.append(actor_loss.detach().item())
            critic_losses.append(critic_loss.detach().item())
            losses.append(loss.detach().item())
            accs.append(acc.detach().item())
            hard_accs.append(hard_acc.detach().item())
        return np.mean(actor_losses), np.mean(critic_losses), np.mean(losses), np.mean(accs), np.mean(hard_accs)

    def optimize(self, step_size, sample_batch_size=0, mini_batch_size=None, interval=0., max_step=None):
        step = 0

        while True:
            steps = sample_steps(n=sample_batch_size)
            if not steps:
                continue
            actor_loss, critic_loss, loss, acc, hard_acc = self.learn_on_steps(steps=steps,
                                                                               mini_batch_size=mini_batch_size)
            # logger.info(f"acc={acc:2.3f}, hard_acc:{hard_acc:2.3f} loss:{loss:2.3f}")

            step += 1
            if step % step_size == 0:
                model_path = self.context.ckpt_model_path(ckpt=step)
                logger.info(f"saving model with ckpt:{step}")
                self.save_model(path=model_path)
                self.conn.send((step, model_path))
                sleep(interval)
                self.conn.recv()
                # logger.info(win_rate)

                # logger.info(f"current lr:{self.schedule.get_last_lr()[0]:2.6f}")
            if max_step and step >= max_step:
                break
            self.schedule.step()

    @ensure_file_path
    def save_model(self, path):
        torch.save(self.ac_model, path)

    def run(self):
        return self.optimize(**self.run_kwargs)


class EvaluateProcess(BaseProcess):
    def __init__(self, env_cls, perfect_steps: List[Step], conn: Connection, best_model,
                 perfect_agent=None, *args, **kwargs):
        super(EvaluateProcess, self).__init__(*args, **kwargs)
        self.env_cls = env_cls
        self.best_acc = 0.
        self.best_model = best_model
        self.perfect_steps = perfect_steps
        self.perfect_agent = perfect_agent
        self.conn = conn

    def _eval_model(self, model):
        if not self.perfect_steps:
            return False
        eval_info = eval_ac_model(ac_model=model, steps=self.perfect_steps)
        with torch.no_grad():
            acc, hard_acc, loss, actor_loss, critic_loss = eval_model_on_steps(model, self.perfect_steps)
            acc, hard_acc, loss, actor_loss, critic_loss = acc.item(), hard_acc.item(), loss.item(), actor_loss.item(), critic_loss.item()
            delta_acc = acc - self.best_acc
            logger.info(
                f"model_acc:{acc:2.3f}({delta_acc:+2.3f}), hard_acc:{hard_acc:2.3f}, loss:{eval_info['loss']:2.3f},"
                f" actor_loss:{eval_info['actor_loss']:2.3f}, critic_loss:{eval_info['critic_loss']:2.3f}")
            self.best_acc = max(acc, self.best_acc)
        return delta_acc > 0

    def _eval_agent(self, model, simulate_num):
        mcst_agent = MCSTAgent(name="evaluate_mcst_agent", ac_model=model, env_cls=self.env_cls,
                               simulate_num=simulate_num)
        steps = self.perfect_steps[:10]
        agent_probs = []
        tgt_probs = torch.from_numpy(np.array([s.probs for s in steps])).float()

        for step in tqdm(steps):
            action_idx, prob, probs = mcst_agent.choose_action(obs=step.next_obs, mode="test")
            agent_probs.append(probs)

        agent_probs = torch.Tensor(agent_probs).float()
        logger.info(agent_probs.shape)

        acc = ((tgt_probs > 0).int() * agent_probs).sum(axis=-1).mean().item()
        hard_probs = torch.nn.functional.one_hot(torch.argmax(agent_probs, dim=-1),
                                                 num_classes=agent_probs.shape[-1]).float()
        hard_acc = ((tgt_probs > 0).int() * hard_probs).sum(dim=-1).mean().item()
        logger.info(f"acc:{acc:2.3f}, hard_acc:{hard_acc:2.3f}")

    def _compare_with_cur_best(self, model, simulate_num, episodes, threshold=0.5):
        logger.info("comparing model with current best model")
        best_mcst_agent = MCSTAgent(name="best_mcst_agent", ac_model=self.best_model, env_cls=self.env_cls,
                                    simulate_num=simulate_num)
        cur_mcst_agent = MCSTAgent(name="current_mcst_agent", ac_model=model, env_cls=self.env_cls,
                                   simulate_num=simulate_num)
        win_rate = compare_agents(agent=cur_mcst_agent, tgt_agent=best_mcst_agent, env=self.env_cls(),
                                  episodes=episodes)

        return win_rate >= threshold

    def _compare_with_perfect(self, model, simulate_num, episodes, threshold=0.5):
        if self.perfect_agent:
            logger.info("comparing model with perfect agent")
            cur_mcst_agent = MCSTAgent(name="current_mcst_agent", ac_model=model, env_cls=self.env_cls,
                                       simulate_num=simulate_num)
            win_rate = compare_agents(agent=cur_mcst_agent, tgt_agent=self.perfect_agent, env=self.env_cls(),
                                      episodes=episodes)

            return win_rate >= threshold

    def evaluate(self, **kwargs):
        while True:
            ckpt, model_path = self.conn.recv()
            cur_model = torch.load(model_path)
            logger.info(f"evaluating ckpt:{ckpt}...")
            # self._eval_agent(model=cur_model, **kwargs)
            self._eval_model(model=cur_model)
            tag = self._compare_with_cur_best(model=cur_model, **kwargs)
            self._compare_with_perfect(model=cur_model, **kwargs)
            if tag:
                logger.info("update best model")
                self.best_model = cur_model
                logger.info(f"save ckpt:{ckpt} to {self.context.best_model_path}")
                self.save_best_model(path=self.context.best_model_path)
                add_ckpt(ckpt, model_path)
            self.conn.send(ckpt)

    @ensure_file_path
    def save_best_model(self, path):
        torch.save(self.best_model, path)

    def run(self):
        return self.evaluate(**self.run_kwargs)


def new_ac_model(env: BoardEnv, model_type, torch_kwargs, ckpt_path=None):
    if ckpt_path:
        logger.info(f"loading model from {ckpt_path}")
        ac_model = torch.load(ckpt_path)
    else:
        ac_model_cls = get_model_cls(env.__class__, model_type)
        ac_model = ac_model_cls(**torch_kwargs, action_num=env.action_num, board_size=env.board_size)
    return ac_model


class ReinforcementLearning:

    def __init__(self, name, config: dict, env, overwrite_time=True):
        self.name = name
        self.config = config
        self.env = env
        self.env_cls = env.__class__
        self.action_num = env.action_num
        base_dir = os.path.join(f"{cur_dir}/../experiments", self.env_cls.__name__, self.name)
        if not overwrite_time:
            base_dir = os.path.join(base_dir, get_current_datetime_str())
        # logger.info(f"base_dir:{base_dir}")
        self.context = Context(base_dir=base_dir)

    def _get_ac_model(self, model_type, torch_kwargs, ckpt_path=None):
        if ckpt_path:
            ac_model = torch.load(ckpt_path)
        else:
            ac_model_cls = get_model_cls(self.env_cls, model_type)
            ac_model = ac_model_cls(**torch_kwargs, action_num=self.action_num, board_size=self.env_cls.board_size)
        return ac_model

    def run(self):
        jdump(self.config, os.path.join(self.context.base_dir, "config.json"))
        ac_model = new_ac_model(env=self.env, **self.config["ac_model_kwargs"])
        mcst_agent = MCSTAgent(name="mcst_agent", env_cls=self.env_cls,
                               ac_model=ac_model, **self.config["mcst_kwargs"])

        #
        # logger.info("start services")
        # cmd = f"python run_services.py --capacity={self.config['buffer_kwargs']['capacity']}"
        # execute_cmd(cmd)

        logger.info("resetting buffer")
        logger.info(reset_buffer(**self.config["buffer_kwargs"]))

        logger.info("generating perfect steps")
        minmax_agent = MinMaxAgent(name="minmax_agent", action_num=self.action_num)
        minmax_agent.train(env=self.env)
        perfect_steps = minmax_agent.gen_steps(env_cls=self.env_cls, step_num=None)

        processes = []
        if self.config.get("supervised", False):
            resp = puts_steps(steps=perfect_steps)
            logger.info(f"current_buffer:{resp['data']}")

        else:
            self_play_process = SelfPlayProcess(name="self_play_process", agent=mcst_agent,
                                                env=self.env, context=self.context,
                                                run_kwargs=self.config["self_play_kwargs"])
            processes.append(self_play_process)

        conn_eval, conn_opt = Pipe()
        optimize_process = OptimizeProcess(name="optimize_process", ac_model=ac_model, conn=conn_opt,
                                           context=self.context,
                                           opt_kwargs=self.config["opt_kwargs"],
                                           schedule_kwargs=self.config["schedule_kwargs"],
                                           run_kwargs=self.config["optimize_kwargs"]
                                           )
        evaluate_process = EvaluateProcess(name="evaluate_process", perfect_steps=perfect_steps, env_cls=self.env_cls,
                                           conn=conn_eval, best_model=copy.deepcopy(ac_model),
                                           perfect_agent=minmax_agent,
                                           context=self.context, run_kwargs=self.config["evaluate_kwargs"])

        processes.append(optimize_process)
        processes.append(evaluate_process)

        for process in processes:
            process.start()
            logger.info(f"process {process.name} starts")

        optimize_process.join()
        logger.info(f"{optimize_process.name} ends")

        for process in processes:
            if process.is_alive():
                logger.info(f"terminating process:{process.name}")
                process.terminate()

        logger.info("all processes done")

    def run_self_play(self, load_model=True, ckpt=None):
        if load_model:
            if ckpt:
                model_path = self.context.ckpt_model_path(ckpt=ckpt)
            else:
                model_path = self.context.best_model_path
            logger.info(f"loading model from {model_path}")
            ac_model = torch.load(model_path)
        else:
            ac_model = self._get_ac_model(**self.config["ac_model_kwargs"])
        mcst_agent = MCSTAgent(name="self_play_mcst_agent", env_cls=self.env_cls,
                               ac_model=ac_model, **self.config["mcst_kwargs"])

        logger.info("resetting buffer")
        logger.info(reset_buffer(**self.config["buffer_kwargs"]))

        self_play_process = SelfPlayProcess(name="self_play_process", agent=mcst_agent,
                                            env=self.env, context=self.context,
                                            run_kwargs=self.config["self_play_kwargs"])

        self_play_process.start()
        logger.info("self play starts")
        self_play_process.join()
        logger.info("self play finished")
