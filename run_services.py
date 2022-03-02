#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_services.py
   Author :       chenhao
   time：          2022/2/22 15:30
   Description :
-------------------------------------------------
"""
import click
import logging
from flask import Flask, request, jsonify

from rlb.core import Step
from rlb.utils import ReplayBuffer

app = Flask(__name__)
logger = logging.getLogger(__name__)

buffer = ReplayBuffer(capacity=100)
ckpts = []


@app.route('/add_ckpt', methods=["POST"])
def add_ckpt():
    ckpt_info = request.json
    ckpts.append(ckpt_info)
    resp = dict(result="success", data=dict(length=len(ckpts), **ckpt_info))
    return jsonify(resp)


@app.route('/get_ckpt', methods=["GET"])
def get_ckpt():
    if not ckpts:
        resp = dict(result="fail", data=dict(length=len(ckpts), ckpt_info=None))
    else:
        ckpt_info = ckpts[-1]
        resp = dict(result="success", data=dict(length=len(ckpts), **ckpt_info))
    return jsonify(resp)


@app.route('/puts', methods=["POST"])
def puts():
    content = request.json
    steps = [Step(**e) for e in content]
    buffer.puts(steps)
    resp = dict(result="success", data=dict(length=len(buffer), acc_idx=buffer.acc_idx, added=len(steps)))
    # logger.info(resp)
    return jsonify(resp)


@app.route('/sample/<int:n>')
def sample(n):
    if n == 0:
        n = len(buffer)
    if len(buffer) < n:
        logger.info(f"{len(buffer)}<{n}, will not sample")
        resp = dict(result="fail", data=[])
    else:
        steps = buffer.sample(n)
        resp = dict(result="success", data=[e.dict() for e in steps])
    return jsonify(resp)


@app.route('/reset/<int:capacity>')
def reset(capacity):
    buffer.clear()
    ckpts.clear()
    resp = dict(result="success", data=dict(buffer_len=len(buffer), ckpt_len=len(ckpts)))
    buffer.capacity = capacity
    return jsonify(resp)


@click.command()
@click.option('--capacity', default=100, help='capacity')
def start_buffer_service(capacity):
    buffer.capacity = capacity
    logger.info("starting buffer service")
    app.run(debug=False)


if __name__ == "__main__":
    start_buffer_service()
