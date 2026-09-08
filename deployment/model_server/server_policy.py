# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import os
import socket

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def main(args) -> None:
    """Build the policy wrapper and start the websocket server.

    The wrapper now owns un-normalization + chunk_size discovery so that all
    eval clients (LIBERO / SimplerEnv / etc.) just need to forward `examples`
    and consume already-unnormalized actions from the response.
    """
    # Set this before importing the framework so its global CUDA backend
    # settings can be selected before torch modules are initialized.
    if args.fast_inference:
        os.environ["STARVLA_FAST_INFERENCE"] = "1"

    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
        enable_world_model=args.enable_world_model,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server; wrapper.metadata is sent at handshake.
    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )
    logging.info(
        "server running ... max_batch_size=%d batch_wait_ms=%.1f metadata=%s",
        args.max_batch_size,
        args.batch_wait_ms,
        wrapper.metadata,
    )
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument("--enable_world_model", action="store_true", help="Enable Fast-WAM world model feature injection")
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Maximum number of queued client requests per model inference batch.",
    )
    parser.add_argument(
        "--batch_wait_ms",
        type=float,
        default=5.0,
        help="Maximum time to wait for more compatible requests before running a batch.",
    )
    parser.add_argument(
        "--fast_inference",
        action="store_true",
        help="Enable non-deterministic CUDA backend settings optimized for inference throughput.",
    )
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
