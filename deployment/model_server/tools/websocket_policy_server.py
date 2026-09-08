# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
import traceback
from collections import deque

import numpy as np

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # Idle timeout in seconds, -1 means never auto-close
        metadata: dict | None = None,
        max_batch_size: int = 8,
        batch_wait_ms: float = 5.0,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._max_batch_size = max(1, int(max_batch_size))
        self._batch_wait_ms = max(0.0, float(batch_wait_ms))
        self._last_active = time.time()
        self._infer_queue: asyncio.Queue[_InferenceRequest] | None = None
        self._pending_infer: deque[_InferenceRequest] = deque()
        self._batch_count = 0
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        # The model already accepts a list of examples.  Keep one inference
        # worker per server so CUDA execution remains serialized, while
        # requests from the eight LIBERO clients are coalesced into batches.
        self._infer_queue = asyncio.Queue()
        worker = asyncio.create_task(self._inference_worker())
        try:
            async with websockets.asyncio.server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                ping_interval=None,   # disable keepalive pings (inference can be slow)
                ping_timeout=None,    # disable ping timeout
                close_timeout=60,     # generous close timeout
            ) as server:
                if self._idle_timeout > 0:
                    await self._idle_watchdog(server)
                else:
                    await server.serve_forever()
        finally:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    async def _idle_watchdog(self, server):
        """Monitor idle time and shut down the server on timeout."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # Refresh active time on each received message
                mtype = msg.get("type", "infer") if isinstance(msg, dict) else "infer"
                if mtype in ("infer", "predict_action"):
                    ret = await self._submit_inference(msg)
                else:
                    ret = self._route_message(msg)  # route control message
                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    @staticmethod
    def _error_response(req_id: str, message: str) -> dict:
        return {
            "status": "error",
            "ok": False,
            "type": "inference_result",
            "request_id": req_id,
            "error": {"message": message},
        }

    @staticmethod
    def _batch_signature(payload: dict) -> tuple:
        """Return a stable key for kwargs that must match within a batch."""
        signature = []
        for key, value in payload.items():
            if key == "examples":
                continue
            if isinstance(value, (str, int, float, bool, type(None))):
                encoded = value
            else:
                encoded = repr(value)
            signature.append((key, encoded))
        return tuple(sorted(signature))

    async def _submit_inference(self, msg: dict) -> dict:
        req_id = msg.get("request_id", "default")
        payload = msg.get("payload", msg)
        if not isinstance(payload, dict):
            return self._error_response(req_id, "Payload must be a dict")

        examples = payload.get("examples")
        if not isinstance(examples, list) or not examples:
            return self._error_response(req_id, "Inference payload must contain a non-empty examples list")

        if self._infer_queue is None:
            return self._error_response(req_id, "Inference queue is not ready")

        future = asyncio.get_running_loop().create_future()
        request = _InferenceRequest(req_id=req_id, payload=payload, future=future)
        await self._infer_queue.put(request)
        return await future

    async def _next_inference_request(self) -> "_InferenceRequest":
        if self._pending_infer:
            return self._pending_infer.popleft()
        if self._infer_queue is None:
            raise RuntimeError("Inference queue is not ready")
        return await self._infer_queue.get()

    async def _collect_inference_batch(self) -> list["_InferenceRequest"]:
        first = await self._next_inference_request()
        batch = [first]
        signature = self._batch_signature(first.payload)

        if self._max_batch_size == 1 or self._batch_wait_ms <= 0:
            return batch

        deadline = asyncio.get_running_loop().time() + self._batch_wait_ms / 1000.0
        while len(batch) < self._max_batch_size:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                if self._infer_queue is None:
                    break
                candidate = await asyncio.wait_for(self._infer_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break

            if self._batch_signature(candidate.payload) == signature:
                batch.append(candidate)
            else:
                # Different inference kwargs (e.g. a different unnorm_key)
                # must not be mixed with this batch.  Keep it for the next one.
                self._pending_infer.append(candidate)

        return batch

    async def _inference_worker(self) -> None:
        while True:
            batch = await self._collect_inference_batch()
            started = time.perf_counter()

            merged_payload = dict(batch[0].payload)
            merged_examples = []
            request_sizes = []
            for request in batch:
                request_examples = request.payload["examples"]
                merged_examples.extend(request_examples)
                request_sizes.append(len(request_examples))
            merged_payload["examples"] = merged_examples

            try:
                # One worker serializes CUDA execution.  The LIBERO client has
                # one outstanding request per connection, so blocking here is
                # intentional: it avoids thread dispatch overhead and keeps
                # the model on one CUDA execution thread.
                output = self._policy.predict_action(**merged_payload)
                if not isinstance(output, dict) or "actions" not in output:
                    raise KeyError("Policy output must be a dict containing 'actions'")

                actions = np.asarray(output["actions"])
                total_examples = sum(request_sizes)
                if actions.ndim == 0 or actions.shape[0] != total_examples:
                    raise ValueError(
                        f"Policy returned actions with shape {actions.shape}; "
                        f"expected first dimension {total_examples}"
                    )

                offset = 0
                for request, request_size in zip(batch, request_sizes):
                    end = offset + request_size
                    data = dict(output)
                    data["actions"] = actions[offset:end]
                    response = {
                        "status": "ok",
                        "ok": True,
                        "type": "inference_result",
                        "request_id": request.req_id,
                        "data": data,
                    }
                    if not request.future.cancelled():
                        request.future.set_result(response)
                    offset = end
            except Exception as exc:
                logging.exception(
                    "Policy inference error for batch_size=%d", sum(request_sizes)
                )
                for request in batch:
                    if not request.future.cancelled():
                        request.future.set_result(self._error_response(request.req_id, str(exc)))

            self._batch_count += 1
            if self._batch_count == 1 or self._batch_count % 25 == 0:
                elapsed = time.perf_counter() - started
                logging.info(
                    "Batched inference #%d: requests=%d examples=%d elapsed=%.3fs queue=%d",
                    self._batch_count,
                    len(batch),
                    sum(request_sizes),
                    elapsed,
                    self._infer_queue.qsize() if self._infer_queue is not None else -1,
                )

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


@dataclasses.dataclass
class _InferenceRequest:
    req_id: str
    payload: dict
    future: asyncio.Future


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
