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

from rlb.core import ACInfo
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
    ac_infos = [ACInfo(**e) for e in content]
    buffer.puts(ac_infos)
    resp = dict(result="success", data=dict(length=len(buffer), acc_idx=buffer.acc_idx, added=len(ac_infos)))
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
        ac_infos = buffer.sample(n)
        resp = dict(result="success", data=[e.dict() for e in ac_infos])
    return jsonify(resp)


@app.route('/gets', methods=["POST"])
def gets():
    get_info = request.json
    n = get_info["n"]
    idx = get_info["idx"] % len(buffer)
    end_idx = idx + n
    logger.info(f"getting ac_info from:{idx} to {end_idx}")
    ac_infos = []
    while end_idx > len(buffer):
        ac_infos += buffer.items[idx:]
        idx = 0
        end_idx -= len(buffer)

    ac_infos += buffer.items[idx:end_idx]
    resp = dict(result="success", data=dict(items=[e.dict() for e in ac_infos], idx=end_idx))
    return jsonify(resp)


@app.route('/reset/<int:capacity>')
def reset(capacity):
    buffer.clear()
    ckpts.clear()
    resp = dict(result="success", data=dict(buffer_len=len(buffer), ckpt_len=len(ckpts)))
    buffer.capacity = capacity
    return jsonify(resp)

@app.route('/clear_buffer')
def clear_buffer():
    buffer.clear()
    resp = dict(result="success", data=dict(buffer_len=len(buffer)))
    return jsonify(resp)


@click.command()
@click.option('--capacity', default=100, help='capacity')
def start_buffer_service(capacity):
    buffer.capacity = capacity
    logger.info("starting buffer service")
    app.run(debug=False)


if __name__ == "__main__":
    start_buffer_service()
