import os
import sys
import time
import pickle
import queue
import threading
import concurrent.futures
import dataclasses
from typing import List, Optional, Any
import numpy as np
import zmq
import jax
import flax
from absl import app, flags

# Add src to path if running directly
sys.path.append(os.path.abspath("src"))

from maxdiffusion import pyconfig, max_logging

# --- Constants ---
MAX_CHUNKS = 64
TARGET_SR = 24000
SERVER_ADDRESS = "tcp://0.0.0.0:5555"

# --- Flags ---
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "config",
    "src/maxdiffusion/configs/f5.yml",
    "Path to the F5 config yaml (f5.yml).",
)
flags.DEFINE_integer(
    "max_sequence_length",
    None,
    "Override config.max_sequence_length at startup.",
)
flags.DEFINE_bool(
    "warmup",
    False,
    "Run one warmup inference at startup to pre-compile for the selected batch size and sequence length.",
)
flags.DEFINE_integer(
    "warmup_batch_size",
    1,
    "Warmup request batch size (number of items before server-side padding).",
)
flags.DEFINE_list(
    "warmup_batch_sizes",
    [],
    "Comma-separated list of warmup batch bucket sizes (final padded batch sizes). If set, warmup iterates this list.",
)
flags.DEFINE_integer(
    "warmup_sequence_length",
    None,
    "Warmup max_sequence_length (defaults to runtime config.max_sequence_length).",
)
flags.DEFINE_list(
    "bucket_batch_sizes",
    [],
    "Comma-separated list of inference batch bucket sizes (final padded batch sizes). "
    "If set, inference batches are padded to the smallest bucket >= num_items.",
)
flags.DEFINE_bool(
    "show_timing",
    False,
    "Print timing logs (disabled by default).",
)

# --- Global State ---
global_config = None

global_f5_pipeline = None

@dataclasses.dataclass
class InferenceRequest:
    identity: bytes  # ZMQ identity of the requester
    text: str
    ref_audio: np.ndarray
    duration_frames: int
    ref_text: str

class BatchInferenceManager:
    def __init__(
        self,
        process_func,
        result_callback,
        max_batch_size=64,
        batch_wait_timeout=0.05,
        bucket_batch_sizes: Optional[List[int]] = None,
    ):
        self.process_func = process_func
        self.result_callback = result_callback # Callback to send results back (identity, data)
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout = batch_wait_timeout
        self.bucket_batch_sizes = bucket_batch_sizes or []
        self.queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._process_batch_loop, daemon=True)
        self.worker_thread.start()

    def submit(self, identity: bytes, text: str, ref_audio: np.ndarray, duration_frames: int, ref_text: str):
        req = InferenceRequest(identity, text, ref_audio, duration_frames, ref_text)
        self.queue.put(req)

    def _process_batch_loop(self):
        max_logging.log("BatchInferenceManager worker started.")
        while not self.shutdown_event.is_set():
            batch = []
            try:
                req = self.queue.get(timeout=1.0)
                batch.append(req)
            except queue.Empty:
                continue

            start_wait = time.time()
            while len(batch) < self.max_batch_size:
                time_left = self.batch_wait_timeout - (time.time() - start_wait)
                if time_left <= 0:
                    break
                try:
                    req = self.queue.get(timeout=time_left)
                    batch.append(req)
                except queue.Empty:
                    break
            
            if batch:
                self._execute_batch(batch)

    def _execute_batch(self, batch: List[InferenceRequest]):
        num_items = len(batch)
        max_logging.log(f"Executing batch of {num_items} items.")

        desired_batch_size = _select_batch_bucket(num_items, self.bucket_batch_sizes)
        
        padded_items_count = desired_batch_size - num_items
        
        batched_text = [req.text for req in batch]
        batched_ref_audio = [req.ref_audio for req in batch]
        batched_duration = [req.duration_frames for req in batch]
        
        if padded_items_count > 0:
             pad_req = batch[0]
             for _ in range(padded_items_count):
                 batched_text.append(pad_req.ref_text)
                 batched_ref_audio.append(pad_req.ref_audio)
                 batched_duration.append(batched_duration[0])

        try:
            t0 = time.time()
            audio_out_jax = self.process_func(
                prompt=batched_text,
                reference_audio=batched_ref_audio,
                duration=batched_duration,
                max_sequence_length=global_config.max_sequence_length,
            )
            audio_out_jax.block_until_ready()
            t1 = time.time()
            max_logging.log_timing(f"Batch execution time: {t1-t0:.4f}s")
            audio_out_cpu = np.asarray(audio_out_jax[:num_items])
            
            for i, req in enumerate(batch):
                self.result_callback(req.identity, {"status": "ok", "audio": audio_out_cpu[i]})

        except Exception as e:
            max_logging.error("Error during batch execution", exc_info=True)
            for req in batch:
                self.result_callback(req.identity, {"status": "error", "message": str(e)})


def _compute_default_bucket_size(num_items: int) -> int:
    device_count = jax.device_count()
    multiplier = 1
    while device_count * multiplier < num_items:
        multiplier <<= 1
    return device_count * multiplier


def _select_batch_bucket(num_items: int, bucket_batch_sizes: List[int]) -> int:
    if not bucket_batch_sizes:
        return _compute_default_bucket_size(num_items)

    for bucket in bucket_batch_sizes:
        if bucket >= num_items:
            return bucket
    raise ValueError(
        f"Batch size {num_items} exceeds max bucket {bucket_batch_sizes[-1]}. "
        "Add a larger value to --bucket_batch_sizes."
    )


def _parse_int_list(values: List[str]) -> List[int]:
    parsed: List[int] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        parsed.append(int(s))
    return parsed


def _normalize_bucket_batch_sizes(bucket_batch_sizes: List[int]) -> List[int]:
    normalized = sorted({int(x) for x in bucket_batch_sizes if int(x) > 0})
    return normalized


def _validate_bucket_batch_sizes(bucket_batch_sizes: List[int]) -> None:
    if not bucket_batch_sizes:
        return
    device_count = jax.device_count()
    for b in bucket_batch_sizes:
        if b <= 0:
            raise ValueError(f"Invalid bucket size: {b}. Must be > 0.")
        if device_count > 1 and (b % device_count) != 0:
            raise ValueError(
                f"Invalid bucket size: {b}. Must be divisible by device_count={device_count}."
            )


def _run_warmup(process_func, warmup_bucket_batch_sizes: List[int], warmup_max_sequence_length: int) -> None:
    warmup_bucket_batch_sizes = _normalize_bucket_batch_sizes(warmup_bucket_batch_sizes)
    if not warmup_bucket_batch_sizes:
        max_logging.log("Warmup requested but no warmup buckets resolved.")
        return

    # Make the reference audio "full length" so ref_audio_len spans max_sequence_length.
    target_len_samples = warmup_max_sequence_length * 256 - 256
    ref_audio = np.zeros((target_len_samples,), dtype=np.float32)

    for bucket_size in warmup_bucket_batch_sizes:
        prompts = ["warmup"] * bucket_size
        ref_audios = [ref_audio] * bucket_size
        durations = [warmup_max_sequence_length] * bucket_size

        max_logging.log(
            "Warmup start: "
            f"bucket_size={bucket_size}, "
            f"warmup_max_sequence_length={warmup_max_sequence_length}"
        )
        t0 = time.time()
        audio_out_jax = process_func(
            prompt=prompts,
            reference_audio=ref_audios,
            duration=durations,
            max_sequence_length=warmup_max_sequence_length,
        )
        audio_out_jax.block_until_ready()
        t1 = time.time()
        max_logging.log_timing(f"Warmup bucket {bucket_size} done in {t1 - t0:.4f}s")


def main(argv):
    global global_config, global_f5_pipeline
    
    if FLAGS.show_timing:
        os.environ["F5_SHOW_TIMING"] = "1"

    # --- Setup JAX/Model ---
    config_path = FLAGS.config
    if not os.path.exists(config_path) and FLAGS.config == "src/maxdiffusion/configs/f5.yml":
        config_path = os.path.join(os.path.dirname(__file__), "configs/f5.yml")
    
    # If config not loaded, initialize it (minimal)
    if not pyconfig.config:
        init_argv = ["f5_tts_serving.py", config_path]
        if FLAGS.max_sequence_length is not None:
            init_argv.append(f"max_sequence_length={FLAGS.max_sequence_length}")
        pyconfig.initialize(init_argv)
        flax.config.update("flax_always_shard_variable", False)

    global_config = pyconfig.config
    
    max_logging.log("Loading F5 Pipeline...")
    from maxdiffusion.checkpointing.f5_checkpointer import F5Checkpointer
    checkpoint_loader = F5Checkpointer(global_config, "F5_CHECKPOINT")
    global_f5_pipeline,_,_ = checkpoint_loader.load_checkpoint()

    bucket_batch_sizes = _normalize_bucket_batch_sizes(_parse_int_list(FLAGS.bucket_batch_sizes))
    warmup_batch_sizes = _normalize_bucket_batch_sizes(_parse_int_list(FLAGS.warmup_batch_sizes))

    if warmup_batch_sizes:
        if bucket_batch_sizes and warmup_batch_sizes != bucket_batch_sizes:
            raise ValueError(
                "warmup_batch_sizes must match bucket_batch_sizes to keep inference buckets consistent with warmup."
            )
        if not bucket_batch_sizes:
            bucket_batch_sizes = warmup_batch_sizes

    _validate_bucket_batch_sizes(bucket_batch_sizes)

    if FLAGS.warmup:
        warmup_max_seq_len = (
            int(FLAGS.warmup_sequence_length)
            if FLAGS.warmup_sequence_length is not None
            else int(global_config.max_sequence_length)
        )
        if warmup_max_seq_len != int(global_config.max_sequence_length):
            max_logging.log(
                "Warning: warmup_sequence_length differs from runtime config.max_sequence_length. "
                "Warmup may not cover runtime shapes."
            )
        if warmup_batch_sizes:
            warmup_bucket_batch_sizes = warmup_batch_sizes
        elif bucket_batch_sizes:
            warmup_bucket_batch_sizes = bucket_batch_sizes
        else:
            warmup_bucket_batch_sizes = [_compute_default_bucket_size(int(FLAGS.warmup_batch_size))]
            bucket_batch_sizes = warmup_bucket_batch_sizes

        _run_warmup(global_f5_pipeline, warmup_bucket_batch_sizes, warmup_max_seq_len)
    
    # --- Setup ZMQ ---
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(SERVER_ADDRESS)
    max_logging.log(f"Inference Server listening on {SERVER_ADDRESS}")

    # Result queue to send back to main loop
    result_queue = queue.Queue()

    def send_result_callback(identity, data):
        result_queue.put((identity, data))

    # Initialize Batch Manager
    batch_manager = BatchInferenceManager(
        global_f5_pipeline,
        send_result_callback,
        max_batch_size=MAX_CHUNKS,
        bucket_batch_sizes=bucket_batch_sizes,
    )

    # --- Event Loop ---
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    try:
        while True:
            # Poll for input (timeout 10ms to allow checking result_queue)
            socks = dict(poller.poll(10))
            
            if socket in socks and socks[socket] == zmq.POLLIN:
                # Receive multipart: [identity, empty, data]
                try:
                    parts = socket.recv_multipart()
                    identity = parts[0]
                    # parts[1] is empty delimiter if using REQ/ROUTER standard, but check 
                    # If connecting directly REQ->ROUTER, REQ sends empty frame first? No, REQ prepends empty frame on recv, sends empty on send? 
                    # Let's handle standard [identity, delimiter, data] or [identity, data] depending on how REQ sends.
                    # Standard REQ adds an empty frame. ROUTER sees [identity, empty, data].
                    
                    if len(parts) >= 3 and parts[1] == b'':
                        data_bytes = parts[2]
                    elif len(parts) == 2: # Sometimes just identity and data if manual
                        data_bytes = parts[1]
                    else:
                         # Try the last part as data
                         data_bytes = parts[-1]

                    request_data = pickle.loads(data_bytes)
                    
                    # Push to batch manager
                    batch_manager.submit(
                        identity=identity,
                        text=request_data['text'],
                        ref_audio=request_data['ref_audio'],
                        duration_frames=request_data['duration_frames'],
                        ref_text=request_data['ref_text']
                    )
                except Exception as e:
                    max_logging.error(f"Error receiving/parsing request: {e}")

            # Check for results to send back
            while True:
                try:
                    identity, result_data = result_queue.get_nowait()
                    serialized_result = pickle.dumps(result_data)
                    # Send back: [identity, empty, data]
                    socket.send_multipart([identity, b'', serialized_result])
                except queue.Empty:
                    break
    
    except KeyboardInterrupt:
        max_logging.log("Shutting down...")
    finally:
        batch_manager.shutdown_event.set()
        context.term()

if __name__ == "__main__":
    app.run(main)
