#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     client.py
   Author :       chenhao
   time：          2022/2/23 17:29
   Description :
-------------------------------------------------
"""
import logging
from typing import List

import requests

from rlb.core import Step

logger = logging.getLogger(__name__)

_buffer_url = "http://127.0.0.1:5000"


def puts_steps(steps: List[Step]):
    data = [s.dict() for s in steps]
    resp = requests.post(url=f"{_buffer_url}/puts", json=data)
    resp.raise_for_status()
    # logger.info(resp.json())
    return resp.json()


def sample_steps(n):
    resp = requests.get(url=f"{_buffer_url}/sample/{n}")
    resp.raise_for_status()
    steps = [Step(**e) for e in resp.json()["data"]]
    return steps


def add_ckpt(ckpt, model_path):
    data = dict(ckpt=ckpt, model_path=model_path)
    resp = requests.post(url=f"{_buffer_url}/add_ckpt", json=data)
    resp.raise_for_status()
    return resp.json()["data"]


def get_ckpt():
    resp = requests.get(url=f"{_buffer_url}/get_ckpt")
    resp.raise_for_status()
    return resp.json()["data"]


def reset_buffer(capacity):
    # logger.info(f"{_buffer_url}/reset/{capacity}")
    resp = requests.get(url=f"{_buffer_url}/reset/{capacity}")
    resp.raise_for_status()
    return resp.json()
