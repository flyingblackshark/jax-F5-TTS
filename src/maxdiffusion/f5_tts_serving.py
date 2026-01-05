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

# --- Global State ---
global_config = None
global_max_sequence_length = 2048
global_f5_pipeline = None

@dataclasses.dataclass
class InferenceRequest:
    identity: bytes  # ZMQ identity of the requester
    text: str
    ref_audio: np.ndarray
    duration_frames: int
    ref_text: str

class BatchInferenceManager:
    def __init__(self, process_func, result_callback, max_batch_size=64, batch_wait_timeout=0.05):
        self.process_func = process_func
        self.result_callback = result_callback # Callback to send results back (identity, data)
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout = batch_wait_timeout
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
        
        device_count = jax.device_count()
        multiplier = 1
        while device_count * multiplier < num_items:
            multiplier <<= 1
        desired_batch_size = device_count * multiplier
        
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
            audio_out_jax = self.process_func(
                prompt=batched_text,
                reference_audio=batched_ref_audio,
                duration=batched_duration,
                max_sequence_length=global_max_sequence_length,
            )
            audio_out_jax.block_until_ready()
            audio_out_cpu = np.asarray(audio_out_jax[:num_items])
            
            for i, req in enumerate(batch):
                self.result_callback(req.identity, {"status": "ok", "audio": audio_out_cpu[i]})

        except Exception as e:
            max_logging.error("Error during batch execution", exc_info=True)
            for req in batch:
                self.result_callback(req.identity, {"status": "error", "message": str(e)})

def main(argv):
    global global_config, global_f5_pipeline, global_max_sequence_length
    
    # --- Setup JAX/Model ---
    config_path = "src/maxdiffusion/configs/f5.yml"
    if not os.path.exists(config_path):
         config_path = os.path.join(os.path.dirname(__file__), "configs/f5.yml")
    
    # If config not loaded, initialize it (minimal)
    if not pyconfig.config:
         pyconfig.initialize(["f5_tts_serving.py", config_path])
         flax.config.update('flax_always_shard_variable', False)

    global_config = pyconfig.config
    
    max_logging.log("Loading F5 Pipeline...")
    from maxdiffusion.checkpointing.f5_checkpointer import F5Checkpointer
    checkpoint_loader = F5Checkpointer(global_config, "F5_CHECKPOINT")
    global_f5_pipeline,_,_ = checkpoint_loader.load_checkpoint()
    
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
    batch_manager = BatchInferenceManager(global_f5_pipeline, send_result_callback, max_batch_size=MAX_CHUNKS)

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
