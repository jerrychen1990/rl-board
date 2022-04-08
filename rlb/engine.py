#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     engine.py
   Author :       chenhao
   time：          2022/4/7 14:52
   Description :
-------------------------------------------------
"""
from collections import defaultdict
from typing import Tuple

from snippets import merge_dicts, jdump_lines
from snippets.decorators import LogTimeCost, log_cost_time, set_kwargs
from tqdm import tqdm

from rlb.constant import DRAW
from rlb.core import *

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
        self.pieces = env.pieces
        assert len(agents) == len(self.pieces)
        self.piece2agent = dict(zip(self.pieces, self.agents))

    def _prepare(self, is_render=False, callbacks=[], episode_idx=0):
        on_callbacks(callbacks=callbacks, event="episode_start", episode_idx=episode_idx)
        state = self.env.reset()
        if is_render:
            self.env.render(state)
        return state

    def step(self, agent, state, mode, is_render, callbacks, step_idx, acc_step_idx) -> Step:
        on_callbacks(callbacks=callbacks, event="step_start", step_idx=step_idx, acc_step_idx=acc_step_idx)
        if is_render:
            logger.info(f"agent:{agent}'s action...")
        mask = self.env.get_mask(state)
        if sum(mask) == 0:
            logger.debug(f"agent{agent} has no valid action, force pass")
            action = PASS_ACTION
            prob = 1.
            probs = [0.] * agent.action_num
        else:
            action_info = agent.choose_action(state=state, mode=mode, mask=mask, step_idx=step_idx,
                                              acc_step_idx=acc_step_idx)
            action = self.env.build_action(action_info)
            prob, probs = action_info.prob, action_info.probs

        if is_render:
            logger.info(f"agent:{agent.name} take action:{action} with prob:{prob:1.3f}")
        transfer_info = self.env.transfer(state, action)
        step = Step(agent_name=agent.name, state=state, action=action, prob=prob, probs=probs, **transfer_info.dict())
        if is_render:
            self.env.render(step.next_state)
        on_callbacks(callbacks=callbacks, event="step_end", step_idx=step_idx, acc_step_idx=acc_step_idx, step=step)
        return step

    def _shuffle(self, offset):
        self.piece2agent = {piece: self.agents[(idx + offset) % len(self.agents)]
                            for idx, piece in enumerate(self.pieces)}

    @log_cost_time(level=logging.DEBUG)
    def run_episodes(self, episodes: int, mode: str, max_step=None, is_render=False, is_shuffle=True,
                     callbacks=[]) -> Tuple[List[Episode], dict]:
        history = []
        acc_step_idx = 0
        scoreboard = defaultdict(int)

        for episode_idx in tqdm(range(episodes)):
            on_callbacks(callbacks=callbacks, event="episode_start", episode_idx=episode_idx)
            if is_shuffle:
                self._shuffle(offset=episode_idx)
            episode = self.run(mode=mode, max_step=max_step, is_render=is_render, callbacks=callbacks,
                               episode_idx=episode_idx, acc_step_idx=acc_step_idx)
            win_key = episode.winner if episode.winner else DRAW
            scoreboard[win_key] += 1
            history.append(episode)
            acc_step_idx += len(episode)
            on_callbacks(callbacks=callbacks, event="episode_end", episode_idx=episode_idx, episode=episode)
        scoreboard = dict(sorted(scoreboard.items(), key=lambda x: x[1], reverse=True))
        return history, scoreboard

    def run(self, mode: str, max_step=None, is_render=False, callbacks=[], episode_idx=0, acc_step_idx=0):

        with LogTimeCost(info=f"episode:{episode_idx + 1}", level=logging.DEBUG) as cost:
            state = self._prepare(is_render=is_render, callbacks=callbacks, episode_idx=episode_idx)
            steps = []
            step_idx = 0
            while True:
                agent = self.piece2agent[state.piece]
                step = self.step(agent=agent, state=state, mode=mode, is_render=is_render, callbacks=callbacks,
                                 step_idx=step_idx, acc_step_idx=acc_step_idx)
                steps.append(step)
                state = step.next_state
                step_idx += 1
                acc_step_idx += 1
                if step.is_done or (max_step and step_idx >= max_step):
                    win_piece = step.win_piece
                    winner = str(self.piece2agent.get(win_piece)) if win_piece else None
                    if is_render:
                        if winner:
                            logger.info(f"{winner}[{win_piece}] win")
                        else:
                            logger.info(f"draw game")
                    break
        episode = Episode(steps=steps, cost=cost.cost, winner=winner, win_piece=win_piece)
        return episode


def run_board_games(agents: List[Agent], env: BoardEnv, episodes: int, mode: str,
                    max_step=None, is_shuffle=True, is_render=False, callbacks=[]) -> Tuple[List[Episode], dict]:
    game = BoardGame(agents=agents, env=env)
    history, scoreboard = game.run_episodes(episodes=episodes, mode=mode, max_step=max_step, is_shuffle=is_shuffle,
                                            is_render=is_render, callbacks=callbacks)
    return history, scoreboard


def scoreboard2win_rate(scoreboard: dict, key, against_key):
    draw_score = scoreboard.get(DRAW, 0)
    score = scoreboard.get(str(key), 0)
    tgt_score = scoreboard.get(str(against_key), 0)
    win_rate = (score + draw_score * 0.5) / (score + draw_score + tgt_score)
    return win_rate


def compare_agents(agent: Agent, tgt_agent: Agent, env, episodes, is_shuffle=True):
    history, scoreboard = run_board_games(agents=[agent, tgt_agent], env=env, episodes=episodes, mode="test",
                                          is_shuffle=is_shuffle)
    win_rate = scoreboard2win_rate(scoreboard, agent, tgt_agent)
    return scoreboard, win_rate


def compare_agents_with_detail(agent: Agent, tgt_agent: Agent, env, episodes):
    # logger.info("comparing current agent with perfect agent")
    first_scoreboard, first_win_rate = compare_agents(agent=agent, tgt_agent=tgt_agent,
                                                      env=env, is_shuffle=False, episodes=episodes // 2)
    # logger.info("comparing perfect agent with  current agent")
    second_scoreboard, second_win_rate = compare_agents(agent=tgt_agent, tgt_agent=agent,
                                                        env=env, is_shuffle=False, episodes=episodes // 2)
    second_win_rate = 1 - second_win_rate
    logger.info(f"{first_scoreboard=}, {first_win_rate=:2.3f}, {second_scoreboard=}, {second_win_rate=:2.3f}")
    scoreboard = merge_dicts(first_scoreboard, second_scoreboard, reduce_func=lambda x, y: x + y)
    win_rate = scoreboard2win_rate(scoreboard, agent, tgt_agent)
    return scoreboard, win_rate


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
