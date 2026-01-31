"""
Start both the F5 inference server (ZMQ) and the FastAPI service (uvicorn) with one command.

Usage:
  python -m src.maxdiffusion.f5_start_api_and_serving
  python -m src.maxdiffusion.f5_start_api_and_serving --api-host 0.0.0.0 --api-port 8000
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _wait_for_tcp(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _terminate_process(proc: subprocess.Popen, timeout_s: float = 10.0) -> None:
    _terminate_process_with_signals(proc, first_signal=signal.SIGTERM, timeout_s=timeout_s)


def _terminate_process_with_signals(
    proc: subprocess.Popen,
    first_signal: int,
    timeout_s: float = 10.0,
    first_signal_timeout_s: float = 3.0,
) -> None:
    if proc.poll() is not None:
        return

    if os.name == "posix":
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return

        try:
            os.killpg(pgid, first_signal)
        except ProcessLookupError:
            return
    else:
        proc.send_signal(first_signal)

    try:
        proc.wait(timeout=first_signal_timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "posix":
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        proc.terminate()

    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "posix":
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        proc.kill()

    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Start F5 inference server + API server.")
    parser.add_argument("--api-host", default="0.0.0.0", help="Host for FastAPI/uvicorn.")
    parser.add_argument("--api-port", type=int, default=8000, help="Port for FastAPI/uvicorn.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to F5 config yaml (passed to f5_tts_serving.py --config).",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Override max_sequence_length for both serving and API chunking.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm up the inference server at startup (batch size + sequence length).",
    )
    parser.add_argument(
        "--warmup-batch-size",
        type=int,
        default=1,
        help="Warmup batch size (number of items before server-side padding).",
    )
    parser.add_argument(
        "--warmup-batch-sizes",
        default=None,
        help="Comma-separated list of warmup bucket batch sizes (final padded batch sizes).",
    )
    parser.add_argument(
        "--warmup-sequence-length",
        type=int,
        default=None,
        help="Warmup sequence length (defaults to runtime max_sequence_length).",
    )
    parser.add_argument(
        "--bucket-batch-sizes",
        default=None,
        help="Comma-separated list of inference bucket batch sizes (final padded batch sizes).",
    )
    parser.add_argument(
        "--show-timing",
        action="store_true",
        help="Show timing logs (disabled by default).",
    )
    parser.add_argument(
        "--inference-host",
        default="127.0.0.1",
        help="Host used only for readiness check (TCP connect).",
    )
    parser.add_argument(
        "--inference-port",
        type=int,
        default=5555,
        help="Port used only for readiness check (TCP connect).",
    )
    parser.add_argument(
        "--wait-inference-seconds",
        type=float,
        default=120.0,
        help="Seconds to wait for the inference server port to open before starting the API.",
    )
    parser.add_argument(
        "--no-wait-inference",
        action="store_true",
        help="Start API immediately (skip waiting for inference port).",
    )
    parser.add_argument("--uvicorn-log-level", default="info", help="Uvicorn log level.")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    inference_cmd = [sys.executable, "-u", "src/maxdiffusion/f5_tts_serving.py"]
    if args.config:
        inference_cmd.extend(["--config", args.config])
    if args.max_sequence_length is not None:
        inference_cmd.extend(["--max_sequence_length", str(args.max_sequence_length)])
        env["F5_MAX_SEQUENCE_LENGTH"] = str(args.max_sequence_length)
    if args.bucket_batch_sizes:
        inference_cmd.extend(["--bucket_batch_sizes", args.bucket_batch_sizes])
    if args.show_timing:
        env["F5_SHOW_TIMING"] = "1"
        inference_cmd.append("--show_timing")
    if args.warmup:
        inference_cmd.append("--warmup")
        if args.warmup_batch_sizes:
            inference_cmd.extend(["--warmup_batch_sizes", args.warmup_batch_sizes])
        else:
            inference_cmd.extend(["--warmup_batch_size", str(args.warmup_batch_size)])
        if args.warmup_sequence_length is not None:
            inference_cmd.extend(["--warmup_sequence_length", str(args.warmup_sequence_length)])

    api_cmd = [
        sys.executable,
        "-u",
        "-m",
        "uvicorn",
        "src.maxdiffusion.f5_api:app",
        "--host",
        args.api_host,
        "--port",
        str(args.api_port),
        "--log-level",
        args.uvicorn_log_level,
    ]

    start_new_session = os.name == "posix"
    print(f"[launcher] cwd={repo_root}")
    print(f"[launcher] start inference: {' '.join(inference_cmd)}")
    inference_proc = subprocess.Popen(
        inference_cmd, cwd=str(repo_root), env=env, start_new_session=start_new_session
    )
    api_proc: Optional[subprocess.Popen] = None
    stop_signal = signal.SIGTERM

    try:
        if not args.no_wait_inference:
            ok = _wait_for_tcp(args.inference_host, args.inference_port, args.wait_inference_seconds)
            if not ok:
                print(
                    f"[launcher] inference server not ready after {args.wait_inference_seconds}s: "
                    f"{args.inference_host}:{args.inference_port}",
                    file=sys.stderr,
                )
                return 1

        print(f"[launcher] start api: {' '.join(api_cmd)}")
        api_proc = subprocess.Popen(
            api_cmd, cwd=str(repo_root), env=env, start_new_session=start_new_session
        )

        print("[launcher] running. Press Ctrl+C to stop both.")

        def _signal_handler(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        processes = {"inference": inference_proc, "api": api_proc}

        while True:
            for name, proc in processes.items():
                rc = proc.poll()
                if rc is not None:
                    print(f"[launcher] {name} exited with code {rc}", file=sys.stderr)
                    return rc if rc != 0 else 0
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("[launcher] stopping...")
        stop_signal = signal.SIGINT
        return 0
    finally:
        # Stop API first (it depends on inference).
        if api_proc is not None:
            _terminate_process_with_signals(api_proc, first_signal=stop_signal)
        _terminate_process_with_signals(inference_proc, first_signal=stop_signal)


if __name__ == "__main__":
    raise SystemExit(main())
