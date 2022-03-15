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
import os
from collections import defaultdict
from typing import List, Tuple

from snippets import jdump_lines, merge_dicts
from snippets.decorators import LogTimeCost, log_cost_time, set_kwargs

from rlb.constant import DRAW
from rlb.core import Agent, Episode, Step, Env, BoardEnv

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


class BoardGame:
    def __init__(self, agents: List[Agent], env: BoardEnv):
        self.agents = agents
        self.env = env
        self.env_cls = env.__class__
        self.action_cls = self.env_cls.action_cls
        self.pieces = env.pieces
        assert len(agents) == len(self.pieces)
        self.piece2agent = dict(zip(self.pieces, self.agents))

    def _prepare(self, is_render=False, callbacks=[], episode_idx=0, acc_step_idx=0):
        on_callbacks(callbacks=callbacks, event="episode_start", episode_idx=episode_idx)
        obs = self.env.reset()
        if is_render:
            # logger.info(f"episode:{episode_idx + 1} starts")
            self.env.render()
        return obs

    def _step(self, agent, obs, mode, is_render, callbacks, step_idx, acc_step_idx):
        on_callbacks(callbacks=callbacks, event="step_start", step_idx=step_idx, acc_step_idx=acc_step_idx)
        if is_render:
            logger.info(f"agent:{agent}'s action...")
        action_info = agent.choose_action(obs=obs, mode=mode, step_idx=step_idx, acc_step_idx=acc_step_idx)
        action = self.action_cls.from_idx(action_info.action_idx)
        if is_render:
            logger.info(f"agent:{agent.name} take action:{action} with prob:{action_info.prob:1.3f}")
        transfer_info = self.env.step(action)
        step = Step(agent_name=agent.name, obs=obs, **action_info.dict(), **transfer_info.dict())

        if is_render:
            self.env.render()
        on_callbacks(callbacks=callbacks, event="step_end", step_idx=step_idx,
                     acc_step_idx=acc_step_idx, step=step)
        return step

    def _shuffle(self, offset):
        self.piece2agent = {piece: self.agents[(idx + offset) % len(self.agents)]
                            for idx, piece in enumerate(self.pieces)}

    @log_cost_time(level=logging.DEBUG)
    def run_episodes(self, mode: str, episodes, max_step=None, is_render=False, shuffle=True,
                     callbacks=[], show_episode_size=None) -> Tuple[List[Episode], dict]:
        history = []
        acc_step_idx = 0
        scoreboard = defaultdict(int)

        for episode_idx in range(episodes):
            on_callbacks(callbacks=callbacks, event="episode_start", episode_idx=episode_idx)
            if shuffle:
                self._shuffle(offset=episode_idx)
            episode = self.run(mode=mode, max_step=max_step, is_render=is_render, callbacks=callbacks,
                               episode_idx=episode_idx, acc_step_idx=acc_step_idx)
            win_key = episode.winner if episode.winner else DRAW

            scoreboard[win_key] += 1
            history.append(episode)
            acc_step_idx += episode.step
            for agent in self.agents:
                agent.on_episode_end(episode_idx=episode_idx)
            on_callbacks(callbacks=callbacks, event="episode_end", episode_idx=episode_idx, episode=episode)
            if show_episode_size and (episode_idx + 1) % show_episode_size == 0:
                logger.info(f"{episode_idx + 1}/{episodes} episodes done")

        scoreboard = dict(sorted(scoreboard.items(), key=lambda x: x[1], reverse=True))
        return history, scoreboard

    def run(self, mode: str, max_step=None, is_render=False, callbacks=[], episode_idx=0, acc_step_idx=0):

        with LogTimeCost(info=f"episode:{episode_idx + 1}", level=logging.DEBUG) as cost:
            obs = self._prepare(is_render=is_render, callbacks=callbacks, episode_idx=episode_idx,
                                acc_step_idx=acc_step_idx)
            steps = []
            step_idx = 0
            cur_piece = self.env.pieces[0]

            while True:
                agent = self.piece2agent[cur_piece]
                step = self._step(agent=agent, obs=obs, mode=mode, is_render=is_render, callbacks=callbacks,
                                  step_idx=step_idx, acc_step_idx=acc_step_idx)
                steps.append(step)
                obs = step.next_obs
                extra_info = step.extra_info

                step_idx += 1
                acc_step_idx += 1
                if step.is_done or max_step and step_idx >= max_step:
                    win_piece = extra_info.get("win_piece")
                    winner = str(self.piece2agent.get(win_piece)) if win_piece else None
                    if winner and is_render:
                        logger.info(f"{winner}[{win_piece}] win")
                    break
                cur_piece = step.next_piece
        episode = Episode(steps=steps, cost=cost.cost, winner=winner, win_piece=win_piece)
        return episode


def run_board_games(agents: List[Agent], env: BoardEnv, episodes: int, mode,
                    max_step=None, shuffle=True, is_render=False,
                    show_episode_size=None, callbacks=[]) -> Tuple[List[Episode], dict]:
    game = BoardGame(agents=agents, env=env)
    history, scoreboard = game.run_episodes(episodes=episodes, mode=mode, max_step=max_step, shuffle=shuffle,
                                            is_render=is_render, show_episode_size=show_episode_size,
                                            callbacks=callbacks)
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
        history, scoreboard = run_board_games(agents=[self.agent, self.against_agent], episodes=self.eval_episodes,
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


class RecordCallback(AbsCallback):

    @set_kwargs()
    def __init__(self, record_dir, episode_step_size):
        self.records: List[Episode] = []

    def on_episode_end(self, episode_idx, episode: Episode):
        self.records.append(episode)
        if self.episode_step_size and (episode_idx + 1) % self.episode_step_size == 0:
            path = os.path.join(self.record_dir, f"{episode_idx + 1}.jsonl")
            logger.debug(f"dump record:{episode_idx + 1 - self.episode_step_size}-{episode_idx + 1} to {path}")
            jdump_lines(self.records, path)


def scoreboard2win_rate(scoreboard: dict, key, against_key):
    draw_score = scoreboard.get(DRAW, 0)
    score = scoreboard.get(str(key), 0)
    tgt_score = scoreboard.get(str(against_key), 0)
    win_rate = (score + draw_score * 0.5) / (score + draw_score + tgt_score)
    return win_rate


def compare_agents(agent: Agent, tgt_agent: Agent, env, episodes, shuffle=True):
    history, scoreboard = run_board_games(agents=[agent, tgt_agent], env=env, episodes=episodes, mode="test",
                                          shuffle=shuffle)
    win_rate = scoreboard2win_rate(scoreboard, agent, tgt_agent)
    return scoreboard, win_rate


def compare_agents_with_detail(agent: Agent, tgt_agent: Agent, env, episodes):
    # logger.info("comparing current agent with perfect agent")
    first_scoreboard, first_win_rate = compare_agents(agent=agent, tgt_agent=tgt_agent,
                                                      env=env, shuffle=False, episodes=episodes // 2)
    # logger.info("comparing perfect agent with  current agent")
    second_scoreboard, second_win_rate = compare_agents(agent=tgt_agent, tgt_agent=agent,
                                                        env=env, shuffle=False, episodes=episodes // 2)
    second_win_rate = 1 - second_win_rate
    logger.info(f"{first_scoreboard=}, {first_win_rate=:2.3f}, {second_scoreboard=}, {second_win_rate=:2.3f}")
    scoreboard = merge_dicts(first_scoreboard, second_scoreboard, reduce_func=lambda x, y: x + y)
    win_rate = scoreboard2win_rate(scoreboard, agent, tgt_agent)
    return scoreboard, win_rate
