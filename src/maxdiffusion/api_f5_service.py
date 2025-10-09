"""
FastAPI inference service for F5 TTS pipeline with automatic batch padding
to the nearest power-of-two, refactored to follow JetStream's simple_server
pattern using FastAPI lifespan and StreamingResponse.

Run:
  uvicorn --app-dir src maxdiffusion.api_f5_service:app --host 0.0.0.0 --port 8000

Environment:
  Optional env var F5_CONFIG_PATH to point to a custom YAML config.
  Defaults to src/maxdiffusion/configs/f5.yml

Response:
  StreamingResponse with audio/wav bytes (one combined audio per job).
"""

from __future__ import annotations

import os
import time
from typing import List, Union, Optional, Tuple, Dict

import numpy as np
import soundfile as sf
import librosa
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import threading
import queue
import uuid
from dataclasses import dataclass
import io
import base64
import uvicorn
import jax
from maxdiffusion import pyconfig
from maxdiffusion.checkpointing.f5_checkpointer import F5Checkpointer


class InferRequest(BaseModel):
  prompts: Union[str, List[str]]
  reference_audio: Union[str, List[str]]
  duration: Optional[Union[int, List[int]]] = None
  # Removed max_sequence_length and guidance_scale from request


app = FastAPI(title="F5 TTS Inference API", version="0.1.0")


_GLOBAL_PIPELINE = None
_GLOBAL_CONFIG = None
_DEFAULT_SR = 24000
_MAX_PROMPTS_PER_BATCH = 128

# JetStream-like server uses a single request queue and per-request asyncio.Queue
_PENDING_LOCK = threading.Lock()


@dataclass
class Job:
  id: str
  req: "InferRequest"
  res_queue: "asyncio.Queue[bytes | None]"


def _next_power_of_two(n: int) -> int:
  if n <= 1:
    return 1
  return 1 << (n - 1).bit_length()


def _ensure_list(x: Union[str, int, List[str], List[int]], length: int) -> List:
  # If scalar, repeat to `length`; if list shorter, pad by repeating last
  if isinstance(x, list):
    if len(x) >= length:
      return x[:length]
    return x + [x[-1]] * (length - len(x))
  else:
    return [x for _ in range(length)]

def _default_duration_segments(audio_path: str, sr: int = _DEFAULT_SR) -> int:
  # Rough heuristic consistent with generate_f5_pipeline.py's segmentation of 256 samples per segment.
  # duration is decoder segments count; we add a margin.
  y, sr = librosa.load(audio_path, sr=sr)
  base_segments = int(y.shape[-1] // 256) + 1
  return base_segments + 200  # margin for continuation


@asynccontextmanager
async def lifespan(app: FastAPI):
  global _GLOBAL_PIPELINE, _GLOBAL_CONFIG
  if _GLOBAL_PIPELINE is None:
    default_cfg = os.path.join(os.path.dirname(__file__), "configs", "f5.yml")
    cfg_path = os.environ.get("F5_CONFIG_PATH", default_cfg)
    argv = ["api_f5_service.py", cfg_path]
    pyconfig.initialize(argv)
    _GLOBAL_CONFIG = pyconfig.config
    checkpoint_loader = F5Checkpointer(_GLOBAL_CONFIG, "F5_CHECKPOINT")
    _GLOBAL_PIPELINE = checkpoint_loader.load_checkpoint()

  # Create JetStream-style request queue and record event loop
  app.state.req_queue = queue.Queue()
  app.state.aio_loop = asyncio.get_running_loop()

  # Start worker thread
  threading.Thread(target=_jax_worker_loop, name="jax_worker", args=(app,), daemon=True).start()
  yield
  # No explicit teardown needed; thread is daemon


# Rebind app with lifespan following JetStream style
app = FastAPI(lifespan=lifespan, title="F5 TTS Inference API", version="0.1.0")


def _put_async(loop: asyncio.AbstractEventLoop, q: "asyncio.Queue", item):
  loop.call_soon_threadsafe(q.put_nowait, item)


def _normalize_job(job: Job) -> Tuple[List[str], List[str], List[int], Optional[str]]:
  req = job.req
  prompts = req.prompts if isinstance(req.prompts, list) else [req.prompts]
  if len(prompts) == 0:
    return [], [], [], "prompts must be non-empty"

  if isinstance(req.reference_audio, list):
    ref_audios = req.reference_audio
    if len(ref_audios) == 0:
      return [], [], [], "reference_audio list must be non-empty"
    ref_audio_for_duration = ref_audios[0]
  else:
    ref_audios = [req.reference_audio]
    ref_audio_for_duration = req.reference_audio

  if req.duration is None:
    default_seg = _default_duration_segments(ref_audio_for_duration, _DEFAULT_SR)
    durations = [default_seg for _ in range(len(prompts))]
  else:
    durations = req.duration if isinstance(req.duration, list) else [req.duration]

  ref_audios = _ensure_list(ref_audios, len(prompts))
  durations = _ensure_list(durations, len(prompts))

  return prompts, ref_audios, durations, None


def _jax_worker_loop(app: FastAPI):
  while True:
    batch_jobs: List[Job] = []
    batch_prompts: List[str] = []
    batch_ref_audios: List[str] = []
    batch_durations: List[int] = []
    per_job_meta: List[Tuple[str, int]] = []

    first_job = app.state.req_queue.get()
    jobs_to_consider: List[Job] = [first_job]
    try:
      while True:
        j = app.state.req_queue.get_nowait()
        jobs_to_consider.append(j)
    except Exception:
      pass

    for job in jobs_to_consider:
      prompts_j, refs_j, durs_j, err = _normalize_job(job)
      if err is not None:
        _RETURN_Q.put((job.id, {
          "error": err,
          "batch_original": 0,
          "batch_padded_to": 0,
          "output_audio": ""
        }))
        continue

      if len(batch_prompts) + len(prompts_j) > _MAX_PROMPTS_PER_BATCH:
        if len(batch_prompts) == 0:
          allowed = _MAX_PROMPTS_PER_BATCH
          prompts_j = prompts_j[:allowed]
          refs_j = refs_j[:allowed]
          durs_j = durs_j[:allowed]
        else:
          app.state.req_queue.put(job)
          continue

      batch_prompts.extend(prompts_j)
      batch_ref_audios.extend(refs_j)
      batch_durations.extend(durs_j)
      per_job_meta.append((job.id, len(prompts_j)))
      batch_jobs.append(job)

    total = len(batch_prompts)
    if total == 0:
      continue

    padded_to = _next_power_of_two(total)
    if padded_to > total:
      last_prompt = batch_prompts[-1]
      last_ref = batch_ref_audios[-1]
      last_dur = batch_durations[-1]
      pad_count = padded_to - total
      batch_prompts.extend([last_prompt] * pad_count)
      batch_ref_audios.extend([last_ref] * pad_count)
      batch_durations.extend([last_dur] * pad_count)

    try:
      audios = _GLOBAL_PIPELINE(
          prompt=batch_prompts,
          reference_audio=batch_ref_audios,
          duration=batch_durations,
          max_sequence_length=2048,
      )
      numpy_audios = np.asarray(audios)
      # Trim padded outputs: only keep original (non-padded) samples
      original_total = sum(orig_len for _, orig_len in per_job_meta)
      numpy_audios = numpy_audios[:original_total]

      idx = 0
      for job_id, orig_len in per_job_meta:
        if orig_len == 1:
          combined = numpy_audios[idx]
          idx += 1
        else:
          combined = np.concatenate([numpy_audios[idx + k] for k in range(orig_len)], axis=0)
          idx += orig_len
        buf = io.BytesIO()
        sf.write(buf, combined, samplerate=_DEFAULT_SR, format='WAV')
        audio_bytes = buf.getvalue()
        # Find the job's res_queue and stream bytes, then sentinel None
        for jb in batch_jobs:
          if jb.id == job_id:
            _put_async(app.state.aio_loop, jb.res_queue, audio_bytes)
            _put_async(app.state.aio_loop, jb.res_queue, None)
            break
    except Exception as e:
      err_bytes = b""
      for jb_id, orig_len in per_job_meta:
        for jb in batch_jobs:
          if jb.id == jb_id:
            _put_async(app.state.aio_loop, jb.res_queue, err_bytes)
            _put_async(app.state.aio_loop, jb.res_queue, None)
            break


async def streaming_audio(res_queue: "asyncio.Queue[bytes | None]"):
  while True:
    item = await res_queue.get()
    if item is None:
      return
    yield item


@app.get("/health")
def health():
  return {"status": "ok"}


@app.post("/generate")
async def generate(req: InferRequest):
  assert _GLOBAL_PIPELINE is not None, "Pipeline not initialized"
  res_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue()
  job_id = str(uuid.uuid4())
  app.state.req_queue.put_nowait(Job(id=job_id, req=req, res_queue=res_queue))
  return StreamingResponse(streaming_audio(res_queue), media_type="audio/wav")


def main():
  host = os.environ.get("HOST", "0.0.0.0")
  port = int(os.environ.get("PORT", "8000"))
  uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
  main()