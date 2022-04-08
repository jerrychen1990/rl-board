#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     evaluate.py
   Author :       chenhao
   time：          2022/3/4 16:45
   Description :
-------------------------------------------------
"""
import collections
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from snippets import flat, discard_kwarg
from tqdm import tqdm

from rlb.actor_critic import ActorCriticAgent
from rlb.core import ACInfo, Episode, State
from rlb.engine import compare_agents
from rlb.mcst import MCSTAgent
from rlb.model import ModuleActorCritic
from rlb.utils import format_dict, cross_entropy, kl_div, mse, entropy

logger = logging.getLogger(__name__)
cur_dir = os.path.abspath(os.path.dirname(__file__))


def eval_probs(probs: np.array, tgt_probs: np.array):
    ce = cross_entropy(probs, tgt_probs)
    kl = kl_div(probs, tgt_probs)
    hard_tgt_probs = (tgt_probs > 0).astype(int)
    acc = (probs * hard_tgt_probs).sum(axis=-1).mean()
    return dict(ce=ce, kl=kl, acc=acc)


def eval_values(values: np.array, tgt_values: np.array):
    return dict(mse=mse(values, tgt_values))


def eval_probs_values(probs: np.array, tgt_probs: np.array, values: np.array, tgt_values: np.array) -> dict:
    rs = dict(**eval_probs(probs, tgt_probs), **eval_values(values, tgt_values))
    rs["loss"] = rs["kl"] + rs["mse"]
    return rs


def evaluate_episodes(episodes: List[Episode], perfect_ac_infos: List[ACInfo], show_detail=False):
    steps = flat([e.steps for e in episodes])
    logger.info(f"evaluating {len(episodes)} history episodes with {len(steps)} steps")
    infos = dict()
    for step in steps:
        state = step.state
        action_idx = step.action_idx
        probs = step.probs
        if state not in infos:
            infos[state] = collections.defaultdict(list)
        infos[state][action_idx].append(probs)

    valid_acs = [e for e in perfect_ac_infos if e.probs]
    key_obss = set([e.state for e in valid_acs if e.is_in_path])

    # logger.info(f"{len(infos)} state visited")
    valid_ac_dict = {ac.state: ac.probs for ac in valid_acs}
    cover = len(infos) / len(valid_acs)
    key_cover = len(set(infos.keys()) & key_obss) / len(key_obss)
    obss = list(infos.keys())
    tgt_probs = np.array([valid_ac_dict[o] for o in obss])
    probs = np.array([np.array(list(flat(infos[o].values()))).mean(axis=0) for o in obss])
    eval_info = eval_probs(probs, tgt_probs)
    ent = entropy(probs)
    eval_rs = dict(cover=cover, key_cover=key_cover, step_num=len(steps), obs_num=len(infos), entropy=ent, **eval_info)

    if show_detail:
        infos = sorted(infos.items(), key=lambda x: sum([len(e) for e in x[1].values()]), reverse=True)
        for state, action_dict in infos[:5]:
            logger.info(State.from_obs(state))
            for action, probs in sorted(action_dict.items(), key=lambda x: len(x[1]), reverse=True):
                logger.info(f"action:{action} visited {len(probs)} times")
                for prob in probs[:5]:
                    logger.info(prob)
            logger.info("*" * 50)

    return eval_rs


class Evaluator:
    def __init__(self, env, perfect_ac_infos: List[ACInfo], best_model,
                 perfect_agent=None, *args, **kwargs):
        self.env = env
        self.best_acc = 0.
        self.best_hard_acc = 0.
        self.best_loss = 100.
        self.best_model = best_model
        self.perfect_ac_infos = perfect_ac_infos
        self.perfect_agent = perfect_agent

    @discard_kwarg
    def _eval_model(self, model: ModuleActorCritic):
        if not self.perfect_ac_infos:
            return False

        act_obs, critic_obs, tgt_probs, tgt_values = model.ac_info2arrays(self.perfect_ac_infos)
        with torch.no_grad():
            weights = model.forward_act(torch.from_numpy(act_obs))
            values = model.forward_critic(torch.from_numpy(critic_obs)).numpy()
            probs = F.softmax(weights).numpy()
        eval_info = eval_probs_values(probs, tgt_probs, values, tgt_values)

        acc = eval_info.pop("acc")
        loss = eval_info.pop("loss")
        delta_acc = acc - self.best_acc
        delta_loss = loss - self.best_loss
        self.best_loss = min(self.best_loss, loss)
        self.best_acc = max(self.best_acc, acc)
        logger.info(
            f"model eval info:[{acc=:2.3f}[{delta_acc:+2.3f}], {loss=:2.3f}[{delta_loss:+2.3f}]], {format_dict(eval_info)}")
        return delta_loss < 0

    def _eval_agent(self, model, simulate_num):
        mcst_agent = MCSTAgent(name="evaluate_mcst_agent", ac_model=model, env=self.env,
                               simulate_num=simulate_num)
        steps = self.perfect_ac_infos[:10]
        agent_probs = []
        tgt_probs = torch.from_numpy(np.array([s.probs for s in steps])).float()

        for step in tqdm(steps):
            action_idx, prob, probs = mcst_agent.choose_action(state=step.next_state, mode="test")
            agent_probs.append(probs)

        agent_probs = torch.Tensor(agent_probs).float()
        logger.info(agent_probs.shape)

        acc = ((tgt_probs > 0).int() * agent_probs).sum(axis=-1).mean().item()
        hard_probs = torch.nn.functional.one_hot(torch.argmax(agent_probs, dim=-1),
                                                 num_classes=agent_probs.shape[-1]).float()
        hard_acc = ((tgt_probs > 0).int() * hard_probs).sum(dim=-1).mean().item()
        logger.info(f"acc:{acc:2.3f}, hard_acc:{hard_acc:2.3f}")

    @discard_kwarg
    def _compare_with_cur_best(self, model, episodes, agent_type="mcst", threshold=0.5, agent_kwargs={}):
        logger.info(f"compare current agent with best agent, {agent_type=}")
        if agent_type == "mcst":
            best_agent = MCSTAgent(name="best_mcst_agent", ac_model=self.best_model, env=self.env,
                                   **agent_kwargs)
            cur_agent = MCSTAgent(name="current_mcst_agent", ac_model=model, env=self.env, **agent_kwargs)
        elif agent_type == "ac":
            best_agent = ActorCriticAgent(name="best_ac_agent", ac_model=self.best_model, env=self.env,
                                          **agent_kwargs)
            cur_agent = ActorCriticAgent(name="current_ac_agent", ac_model=model, env=self.env, **agent_kwargs)
        else:
            raise ValueError(f"invalid agent_type:{agent_type}")

        scoreboard, win_rate = compare_agents(agent=cur_agent, tgt_agent=best_agent, env=self.env,
                                              episodes=episodes)
        logger.info(f"{scoreboard=}, {win_rate=:2.3f}")

        return win_rate >= threshold

    def _compare_mcst_with_ac(self, model, episodes, threshold=0.5, agent_kwargs={}, **kwargs):
        logger.info(f"compare current ac agent with current mcst_agent")
        mcst_agent = MCSTAgent(name="current_mcst_agent", ac_model=model, env=self.env, **agent_kwargs)
        ac_agent = ActorCriticAgent(name="current_ac_agent", ac_model=model, env=self.env, **agent_kwargs)
        scoreboard, win_rate = compare_agents(agent=mcst_agent, tgt_agent=ac_agent, env=self.env,
                                              episodes=episodes)
        logger.info(f"{scoreboard=}, {win_rate=:2.3f}")

        return win_rate >= threshold

    def _compare_with_perfect(self, model, episodes, agent_type="mcst", threshold=0.5, agent_kwargs={}):
        logger.info(f"compare current agent with perfect agent,{agent_type=}")
        if agent_type == "mcst":
            cur_agent = MCSTAgent(name="current_mcst_agent", ac_model=model, env=self.env, **agent_kwargs)
        elif agent_type == "ac":
            cur_agent = ActorCriticAgent(name="current_ac_agent", ac_model=model, env=self.env, **agent_kwargs)
        else:
            raise ValueError(f"invalid agent_type:{agent_type}")

        scoreboard, win_rate = compare_agents(agent=cur_agent, tgt_agent=self.perfect_agent, env=self.env,
                                              episodes=episodes)
        logger.info(f"{scoreboard=}, {win_rate=:2.3f}")

        return win_rate >= threshold

    # def do_evaluate_model(self, cur_model, **kwargs):
    #     tag = self._eval_model(model=cur_model)
    #     # self._compare_mcst_with_ac(model=cur_model, **kwargs)
    #     tag = self._compare_with_cur_best(model=cur_model, **kwargs)
    #     if self.perfect_agent:
    #         tmp_kwargs = copy.copy(kwargs)
    #         for agent_type in ["mcst", "ac"][:0]:
    #             tmp_kwargs["agent_type"] = agent_type
    #             self._compare_with_perfect(model=cur_model, **tmp_kwargs)
    #     return tag

    def evaluate_episodes(self, episodes: List[Episode], show_detail=False):
        if self.perfect_ac_infos:
            eval_info = evaluate_episodes(episodes=episodes, show_detail=show_detail,
                                          perfect_ac_infos=self.perfect_ac_infos)
            logger.info(f"episodes eval info: {format_dict(eval_info)}")

    def evaluate(self, model: ModuleActorCritic, eval_info: dict, agent_kwargs: dict):
        _eval_func_map = {
            "eval_model": self._eval_model,
            "compare": self._compare_with_cur_best
        }

        tag = True
        for eval_func_name, (matter, kwargs) in eval_info.items():
            eval_func = _eval_func_map[eval_func_name]
            tmp_tag = eval_func(model=model, agent_kwargs=agent_kwargs, **kwargs)
            if matter:
                tag = tag and tmp_tag

        return tag

    # def evaluate(self, conn, **kwargs):
    #     while True:
    #         ckpt, model_path = conn.recv()
    #         cur_model = torch.load(model_path)
    #         logger.info(f"evaluating ckpt:{ckpt}...")
    #         # self._eval_agent(model=cur_model, **kwargs)
    #         tag = self.do_evaluate_model(cur_model, **kwargs)
    #         if tag:
    #             logger.info("update best model")
    #             self.best_model = cur_model
    #             logger.debug(f"save ckpt:{ckpt} to {self.context.best_model_path}")
    #             save_torch_model(model=self.best_model, path=self.context.best_model_path)
    #             add_ckpt(ckpt, model_path)
    #         conn.send(ckpt)
