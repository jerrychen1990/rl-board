#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils.py
   Author :       chenhao
   time：          2022/2/15 15:38
   Description :
-------------------------------------------------
"""
import datetime
import logging
from typing import List

import numpy as np
import torch
from snippets import ensure_file_path

logger = logging.getLogger(__name__)


def weights2probs(weights: np.ndarray, deterministic=False):
    if deterministic:
        max_weight = np.max(weights)
        weights = (weights == max_weight).astype(float)
    else:
        if np.min(weights) < 0:
            logger.info(weights)
            raise Exception(f"min of weights should be greater than 0!")
        if np.sum(weights) == 0:
            weights += 0.1
    probs = weights / np.sum(weights)
    return probs


def sample_by_probs(probs, a=None):
    try:
        action = np.random.choice(a=np.arange(0, len(probs)), p=probs)
        prob = probs[action]
        if a is not None:
            assert a.shape == probs.shape
            action = a[action]
        return action, prob
    except Exception as e:
        raise e


def sample_by_weights(weights: np.ndarray, deterministic=False):
    probs = weights2probs(weights, deterministic=deterministic)
    return sample_by_probs(probs=probs)


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self._clear()
        self.capacity = capacity

    def __len__(self):
        return self.items.__len__()

    def is_full(self):
        return len(self.items) == self.capacity

    def put(self, item):
        if not self.is_full():
            self.items.append(item)
        else:
            self.items[self.offset] = item
        self.acc_idx += 1
        self.offset = (self.offset + 1) % self.capacity

    def puts(self, items: List):
        for item in items:
            self.put(item)

    def clear(self):
        self._clear()

    def _clear(self):
        self.items = []
        self.offset = 0
        self.acc_idx = 0

    def sample(self, n):
        # logging.info(f"{n}/{len(self)} items sampled")
        idxs = np.random.choice(np.arange(0, len(self)), size=n, replace=False)
        items = [self.items[e] for e in idxs]
        return items


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def get_random_pi(action_num):
    pi = np.random.random(action_num)
    pi /= sum(pi)
    return pi


def add_noise(pi, noise_rate):
    n = len(pi)
    noise = get_random_pi(n)
    return pi * (1 - noise_rate) + noise * noise_rate


def get_current_datetime_str(fmt="%Y-%m-%dT%H:%M:%S"):
    dt = datetime.datetime.now()
    return dt.strftime(fmt=fmt)


@ensure_file_path
def save_torch_model(model, path):
    logger.info(f"saving model to path:{path}")
    torch.save(model, path)


def format_dict(d):
    items = []
    for k, v in d.items():
        if isinstance(v, float):
            items.append(f"{k}={v:2.3f}")
        else:
            items.append(f"{k}={v}")
    return f"[{', '.join(items)}]"

def entropy(probs: np.array, eps=1e-9):
    assert probs.ndim == 2
    detail = -probs * np.log(probs + eps)
    #     print(detail.sum(axis=-1))
    return float(detail.sum(axis=-1).mean())


def cross_entropy(probs, tgt_probs, eps=1e-9):
    assert probs.ndim == 2
    assert probs.shape == tgt_probs.shape

    detail = -tgt_probs * np.log(probs + eps)
    #     print(detail.sum(axis=-1))
    return float(detail.sum(axis=-1).mean())


def kl_div(probs, tgt_probs, eps=1e-9):
    ce = cross_entropy(probs, tgt_probs, eps)
    e = entropy(tgt_probs, eps)
    return ce - e

def mse(value, tgt_value):
    assert value.shape==tgt_value.shape
    return float(np.square(value - tgt_value).mean())


def tuplize(v):
    if isinstance(v, list) or isinstance(v, tuple):
        return tuple([tuplize(e) for e in v])
    return v


def show_cache_info(func):
    cache_info = func.cache_info()
    hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses)
    logger.info(f"{func.__name__}'s cache_info:{cache_info}, hit_rate:{hit_rate:2.3f}")
