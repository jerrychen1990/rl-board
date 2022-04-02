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

from rlb.board_core import ACInfo

logger = logging.getLogger(__name__)

_buffer_url = "http://127.0.0.1:5000"


def puts_ac_infos(ac_infos: List[ACInfo]):
    data = [s.dict() for s in ac_infos]
    resp = requests.post(url=f"{_buffer_url}/puts", json=data)
    resp.raise_for_status()
    # logger.info(resp.json())
    return resp.json()


def sample_ac_infos(n):
    resp = requests.get(url=f"{_buffer_url}/sample/{n}")
    resp.raise_for_status()
    ac_infos = [ACInfo(**e) for e in resp.json()["data"]]
    return ac_infos

def get_ac_infos(idx, n):
    data = dict(idx=idx, n=n)
    resp = requests.post(url=f"{_buffer_url}/gets", json=data)
    resp.raise_for_status()
    json_resp = resp.json()
    next_idx = json_resp["data"]["idx"]
    ac_infos = [ACInfo(**e) for e in json_resp["data"]["items"]]
    return ac_infos, next_idx


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

def clear_buffer():
    resp = requests.get(url=f"{_buffer_url}/clear_buffer")
    resp.raise_for_status()
    return resp.json()
