#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     drl.py
   Author :       chenhao
   time：          2022/2/23 17:15
   Description :
-------------------------------------------------
"""
import copy
import logging
import os
import shutil
import warnings

from rlb.utils import save_torch_model

warnings.simplefilter("ignore", UserWarning)

from multiprocessing import Pipe
from snippets import jdump, get_current_datetime_str, jdump_lines, flat, log_cost_time, LogTimeCost

from rlb.client import reset_buffer, puts_ac_infos
from rlb.core import Context, RandomAgent
from rlb.engine import compare_agents_with_detail
from rlb.evaluate import Evaluator
from rlb.mcst import MCSTAgent
from rlb.minmax import MinMaxAgent
from rlb.model import load_ac_model, build_ac_model
from rlb.optimize import Optimizer
from rlb.self_play import SelfPlayer

logger = logging.getLogger(__name__)
cur_dir = os.path.abspath(os.path.dirname(__file__))


class ReinforcementLearning:

    def __init__(self, name, config: dict, env, base_dir=None, overwrite_time=True):
        self.name = name
        self.config = config
        self.env = env
        self.env_cls = env.__class__
        self.action_num = env.action_num
        if base_dir:
            self.base_dir = base_dir
        else:
            self.base_dir = os.path.join(f"{cur_dir}/../experiments", self.env_cls.__name__, self.name)
            if not overwrite_time:
                self.base_dir = os.path.join(self.base_dir, get_current_datetime_str())
        logger.info(f"{self.base_dir=}")
        self.context = Context(base_dir=self.base_dir)
        self.perfect_agent = MinMaxAgent(name="perfect_agent", action_num=self.action_num)

    def _prepare_agents(self, value_decay):
        ac_model_kwargs = self.config["ac_model_kwargs"]
        ckpt = ac_model_kwargs.pop("ckpt", None)
        if ckpt:
            ac_model = load_ac_model(context=self.context, ckpt=-1)
        else:
            ac_model = build_ac_model(env_cls=self.env_cls, **ac_model_kwargs)
        logger.info(f"ac_model:{ac_model}")

        mcst_agent = MCSTAgent(name="mcst_agent", env_cls=self.env_cls,
                               ac_model=ac_model, **self.config["mcst_kwargs"])

        self.perfect_agent.train(env_cls=self.env_cls, value_decay=value_decay)
        return ac_model, mcst_agent

    def _prepare_components(self, ac_model, mcst_agent, always_update):
        # get kwargs
        optimize_kwargs = self.config["optimize_kwargs"]
        best_model = ac_model

        self_player = SelfPlayer(agent=mcst_agent, env=self.env, context=self.context)

        # whether need evaluate
        if always_update:
            training_model = ac_model
        else:
            training_model = copy.deepcopy(ac_model)
        optimizer = Optimizer(name="inner_optimize_process", ac_model=training_model,
                              context=self.context,
                              opt_kwargs=optimize_kwargs["opt_kwargs"],
                              schedule_kwargs=optimize_kwargs["schedule_kwargs"])

        if self.perfect_agent.is_trained:
            perfect_ac_infos = self.perfect_agent.gen_ac_infos()
            perfect_agent = self.perfect_agent
        else:
            perfect_ac_infos, perfect_agent = None, None

        evaluator = Evaluator(name="evaluate_process", perfect_ac_infos=perfect_ac_infos, context=self.context,
                              env_cls=self.env_cls, best_model=best_model, perfect_agent=perfect_agent)

        return self_player, optimizer, evaluator

    def _init_buffer(self, supervised, capacity=None):
        if supervised:
            perfect_ac_infos = self.perfect_agent.gen_ac_infos()
            capacity = len(perfect_ac_infos)
            logger.info(f"resetting buffer and set {capacity=}")
            reset_buffer(capacity=capacity)
            puts_ac_infos(perfect_ac_infos)
        else:
            logger.info(f"resetting buffer and set {capacity=}")
            assert capacity is not None
            reset_buffer(capacity=capacity)

    def sync_train(self):
        jdump(self.config, os.path.join(self.context.base_dir, "config.json"))
        always_update = self.config.get("always_update", False)
        supervised = self.config.get("supervised", False)
        value_decay = self.config.get("value_decay", 1.)

        self_play_kwargs = self.config["self_play_kwargs"]
        optimize_kwargs = self.config["optimize_kwargs"]
        evaluate_kwargs = self.config["evaluate_kwargs"]

        playing_model, mcst_agent = self._prepare_agents(value_decay=value_decay)
        self_player, optimizer, evaluator = self._prepare_components(playing_model, mcst_agent, always_update)
        training_model = optimizer.ac_model

        epoch_episodes = self_play_kwargs["epoch_episodes"]
        epochs = self_play_kwargs["epochs"]

        optimize_steps = optimize_kwargs["optimize_steps"]
        optimize_batch_size = optimize_kwargs["optimize_batch_size"]
        self._init_buffer(supervised=supervised, capacity=optimize_steps * optimize_batch_size)

        total_episodes = 0
        for epoch in range(epochs):
            logger.info(f"epoch:{epoch + 1}/{epochs}")

            if not supervised:
                with LogTimeCost(info=f"self_play:{epoch + 1}", level=logging.INFO):
                    logger.info("running self play")
                    history, scoreboard = self_player.self_play(episodes=epoch_episodes, value_decay=value_decay,
                                                                show_episode_size=max(epoch_episodes // 5, 1))

                    record_path = os.path.join(self.context.record_dir,
                                               f"{total_episodes}-{total_episodes + len(history)}.jsonl")
                    jdump_lines(history, record_path)
                    ac_infos = flat([e.to_ac_infos(value_decay=value_decay) for e in history])
                    logger.info(f"generated {len(ac_infos)} ac_infos")
                    puts_ac_infos(ac_infos)
                    total_episodes += len(history)

            with LogTimeCost(info=f"optimize:{epoch + 1}", level=logging.INFO):
                logger.info("running optimize")
                optimizer.optimize_one_ckpt(optimize_steps, optimize_batch_size,
                                            actor_loss_type=optimize_kwargs["actor_loss_type"])
                optimizer.schedule.step()
                optimizer.save_model()
                logger.info(f"cache_info:{playing_model.act_and_criticize.cache_info()}")
                playing_model.act_and_criticize.cache_clear()


            with LogTimeCost(info=f"evaluate:{epoch + 1}", level=logging.INFO):
                logger.info(f"running evaluate on ckpt:{optimizer.step}")
                if not supervised:
                    evaluator.evaluate_episodes(episodes=history)
                tag = evaluator.evaluate(model=training_model,
                                         agent_kwargs=self.config["mcst_kwargs"],
                                         eval_info=evaluate_kwargs)
                if tag and not always_update:
                    logger.info("saving best model")
                    save_torch_model(model=training_model, path=self.context.best_model_path)
                    logger.info("updating best model")
                    playing_model.load_state_dict(training_model.state_dict())

        logger.info("saving the model after training")
        save_torch_model(playing_model, path=self.context.best_model_path)

    def async_train(self):

        ac_model, mcst_agent, perfect_agent = self._prepare_train()
        perfect_ac_infos = perfect_agent.gen_ac_infos()

        processes = []
        conn_eval, conn_opt = Pipe()
        stand_alone_optimize = True

        if self.config.get("supervised", False):
            resp = puts_ac_infos(ac_infos=perfect_ac_infos)
            logger.info(f"current_buffer:{resp['data']}")

        else:
            run_kwargs = self.config["self_play_kwargs"]
            if "optimize_kwargs" in run_kwargs:
                optimize_process = Optimizer(name="inner_optimize_process", ac_model=ac_model, conn=conn_opt,
                                             context=self.context,
                                             opt_kwargs=self.config["opt_kwargs"],
                                             schedule_kwargs=self.config["schedule_kwargs"],
                                             run_kwargs=dict())
                run_kwargs["optimize_process"] = optimize_process
                stand_alone_optimize = False

            self_play_process = SelfPlayer(name="self_play_process", agent=mcst_agent,
                                           env=self.env, context=self.context,
                                           run_kwargs=run_kwargs)

            processes.append(self_play_process)

        if stand_alone_optimize:
            optimize_process = Optimizer(name="optimize_process", ac_model=ac_model, conn=conn_opt,
                                         context=self.context,
                                         opt_kwargs=self.config["opt_kwargs"],
                                         schedule_kwargs=self.config["schedule_kwargs"],
                                         run_kwargs=self.config["optimize_kwargs"]
                                         )
            processes.append(optimize_process)

        evaluate_process = Evaluator(name="evaluate_process", perfect_ac_infos=perfect_ac_infos,
                                     env_cls=self.env_cls,
                                     conn=conn_eval, best_model=copy.deepcopy(ac_model),
                                     perfect_agent=perfect_agent,
                                     context=self.context, run_kwargs=self.config["evaluate_kwargs"])

        processes.append(evaluate_process)

        for process in processes:
            process.start()
            logger.info(f"process {process.name} starts")

        for process in processes:
            process.join()
            logger.info(f"process:{process.name} done")

        # optimize_process.join()
        # logger.info(f"{optimize_process.name} ends")
        #
        # for process in processes:
        #     if process.is_alive():
        #         logger.info(f"terminating process:{process.name}")
        #         process.terminate()

        logger.info("all processes done")

    def eval_with_perfect_agent(self, episodes: int):
        assert self.perfect_agent.is_trained
        logger.info("evaluate best agent with perfect agents")
        ac_model = load_ac_model(context=self.context, ckpt=-1)
        mcst_agent = MCSTAgent(name="mcst_agent", env_cls=self.env_cls,
                               ac_model=ac_model, **self.config["mcst_kwargs"])
        scoreboard, win_rate = compare_agents_with_detail(agent=mcst_agent, tgt_agent=self.perfect_agent, env=self.env,
                                                          episodes=episodes)
        logger.info(f"{scoreboard=}, {win_rate=:2.3f}")

    def eval_with_random_agent(self, episodes: int):
        logger.info("evaluate best agent with random agents")
        ac_model = load_ac_model(context=self.context, ckpt=-1)
        mcst_agent = MCSTAgent(name="mcst_agent", env_cls=self.env_cls,
                               ac_model=ac_model, **self.config["mcst_kwargs"])
        random_agent = RandomAgent(name="random_agent", action_num=self.action_num)
        scoreboard, win_rate = compare_agents_with_detail(agent=mcst_agent, tgt_agent=random_agent, env=self.env,
                                                          episodes=episodes)
        logger.info(f"{scoreboard=}, {win_rate=:2.3f}")

    @log_cost_time()
    def train(self):
        sync = self.config.get("sync", False)
        if os.path.exists(self.context.base_dir):
            logger.info(f"clean history under {self.context.base_dir}")
            shutil.rmtree(self.context.base_dir)
        if sync:
            return self.sync_train()
        else:
            return self.async_train()
