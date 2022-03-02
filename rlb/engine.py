#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     engine.py
   Author :       chenhao
   time：          2022/2/15 10:55
   Description :
-------------------------------------------------
"""
import logging
from collections import defaultdict
from typing import List

from snippets.decorators import LogTimeCost, log_cost_time

from rlb.constant import DRAW
from rlb.core import Agent, Episode, Step, Env

logger = logging.getLogger(__name__)


class AbsCallback(object):
    def on_episode_end(self, episode_idx, episode: Episode):
        pass

    def on_episode_start(self, episode_idx):
        pass

    def on_step_start(self, step_idx, acc_step_idx):
        pass

    def on_step_end(self, step_idx, acc_step_idx, step: Step):
        pass

    def on_event(self, event: str, **kwargs):
        if event == "step_start":
            return self.on_step_start(**kwargs)
        if event == "step_end":
            return self.on_step_end(**kwargs)
        if event == "episode_start":
            return self.on_episode_start(**kwargs)
        if event == "episode_end":
            return self.on_episode_end(**kwargs)
        raise Exception(f"invalid event:{event}!")


def on_callbacks(callbacks: List[AbsCallback], event, **kwargs):
    for callback in callbacks:
        callback.on_event(event=event, **kwargs)

@log_cost_time(level=logging.DEBUG)
def run_episodes(agents: List[Agent], env: Env, episodes: int, mode,
                 show_episode_size=None, max_step=None, is_render=False, shuffle=True, callbacks=[]):
    history = []
    acc_step_idx = 0
    scoreboard = defaultdict(int)
    for episode_idx in range(episodes):
        with LogTimeCost(info=f"episode:{episode_idx + 1}", level=logging.DEBUG) as log_cost:
            on_callbacks(callbacks=callbacks, event="episode_start", episode_idx=episode_idx)
            obs = env.reset()
            if is_render:
                env.render()

            steps = []
            winner = DRAW
            step_idx = 0
            agent_idx = episode_idx % len(agents) if shuffle else 0
            while True:
                agent = agents[agent_idx]
                on_callbacks(callbacks=callbacks, event="step_start", step_idx=step_idx, acc_step_idx=acc_step_idx)
                # logging.debug(f"current state:\n{obs}")

                action_info = agent.choose_action(obs=obs, mode=mode, step_idx=step_idx,
                                                  acc_step_idx=acc_step_idx)
                action = env.action_cls.from_idx(action_info.action_idx)
                if is_render:
                    logging.info(f"agent:{agent.name} take action:{action} with prob:{action_info.prob:1.3f}")
                transfer_info = env.step(action)
                is_done = transfer_info.is_done
                step = Step(agent_name=agent.name, obs=obs, **action_info.dict(), **transfer_info.dict())

                steps.append(step)
                if is_render:
                    env.render()
                on_callbacks(callbacks=callbacks, event="step_end", step_idx=step_idx,
                             acc_step_idx=acc_step_idx, step=step)

                step_idx += 1
                acc_step_idx += 1
                obs = transfer_info.next_obs
                if transfer_info.extra_info.get("valid", True):
                    agent_idx = (agent_idx + 1) % len(agents)


                if step.is_done or max_step and step_idx >= max_step:
                    if step.extra_info.get("is_win"):
                        winner = agent
                    break

        scoreboard[winner] += 1
        episode = Episode(steps=steps, cost=log_cost.cost)
        history.append(episode)
        for agent in agents:
            agent.on_episode_end(episode_idx=episode_idx)
        on_callbacks(callbacks=callbacks, event="episode_end", episode_idx=episode_idx, episode=episode)
        if show_episode_size and (episode_idx + 1) % show_episode_size == 0:
            logger.info(f"{episode_idx + 1}/{episodes} episodes done")

    env.close()
    scoreboard = dict(sorted(scoreboard.items(), key=lambda x: x[1], reverse=True))
    return history, scoreboard


class EvalCallback(AbsCallback):
    def __init__(self, agent: Agent, against_agent: Agent, env: Env, eval_episodes: int, episode_step_size=None,
                 acc_step_size=None):
        if not episode_step_size and not acc_step_size:
            raise Exception(f"neither episode_step_size nor acc_step_size are given!")
        self.env = env
        self.agent = agent
        self.against_agent = against_agent
        self.eval_episodes = eval_episodes
        self.episode_step_size = episode_step_size
        self.acc_step_size = acc_step_size

    def _eval(self):
        history, scoreboard = run_episodes(agents=[self.agent, self.against_agent], episodes=self.eval_episodes,
                                           env=self.env, mode="test",
                                           show_episode_size=None, max_step=None, is_render=False, shuffle=True,
                                           callbacks=[])
        logger.info(f"eval scoreboard:{scoreboard}")

    def on_episode_end(self, episode_idx, episode: Episode):
        if self.episode_step_size and (episode_idx + 1) % self.episode_step_size == 0:
            logger.info(
                f"eval {self.agent} against {self.against_agent} with {self.eval_episodes} episodes after {episode_idx + 1} episodes")
            self._eval()

    def on_step_end(self, step_idx, acc_step_idx, step: Step):
        if self.acc_step_size and (acc_step_idx + 1) % self.acc_step_size == 0:
            logger.info(
                f"eval {self.agent} against {self.against_agent} with {self.eval_episodes} episodes after {acc_step_idx + 1} episodes")
            self._eval()
