
import os
import sys
import time
import base64
import io
import pickle
import asyncio
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import zmq
import zmq.asyncio
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(os.path.abspath("src"))

# --- Constants ---
TARGET_SR = 24000
MAX_CHUNKS = 64
MAX_INFERENCE_STEPS = 100
SERVER_ADDRESS = "tcp://127.0.0.1:5555"

# --- Utils ---
# Minimal utils needed for chunking (copied/imported from original if available, but keeping it self contained for now if simple)
# Actually, let's reuse the import as in original to ensure consistency
try:
    from maxdiffusion.utils.pinyin_utils import chunk_text
except ImportError:
    # Fallback or error if not found. Assuming environment is set up.
    # For robust refactor, let's rely on the import.
    sys.path.append(os.path.abspath("src"))
    from maxdiffusion.utils.pinyin_utils import chunk_text


def _timing_enabled() -> bool:
    value = os.environ.get("F5_SHOW_TIMING", "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


# --- Global State ---
zmq_context = None
zmq_socket = None

# --- API Models ---
class GenerateRequest(BaseModel):
    text: str = Field(..., description="Text to generate")
    ref_audio: str = Field(..., description="Base64 encoded reference audio (wav/mp3)")
    ref_text: str = Field(DEFAULT="and there are so many things about humankind that is bad and evil. I strongly believe that love is one of the only things we have in this world.", description="Reference text")
    steps: int = 50
    cfg: float = 2.0
    speed: float = 1.0
    sway_sampling: bool = False
    gen_len: int = Field(default=None, description="Fixed generation length in frames. If set, text chunking is disabled.")

class GenerateResponse(BaseModel):
    audio_base64: str
    sample_rate: int = TARGET_SR

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup ZMQ
    global zmq_context, zmq_socket
    zmq_context = zmq.asyncio.Context()
    zmq_socket = zmq_context.socket(zmq.REQ)
    zmq_socket.connect(SERVER_ADDRESS)
    print(f"API Server connected to Inference Server at {SERVER_ADDRESS}")
    
    yield
    
    # Cleanup
    if zmq_socket:
        zmq_socket.close()
    if zmq_context:
        zmq_context.term()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    # --- Preprocessing (CPU intensive, done here) ---
    t0 = time.time()
    
    # Decode audio
    try:
        audio_data = base64.b64decode(request.ref_audio)
        ref_audio, ref_sr = librosa.load(io.BytesIO(audio_data), sr=TARGET_SR, mono=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio input: {e}")

    if ref_audio.size == 0:
        raise HTTPException(status_code=400, detail="Empty reference audio")

    ref_text = request.ref_text
    gen_text = request.text
    speed_factor = request.speed
    
    # Ensure reference text ends with space if last char is ASCII
    if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    ref_duration_sec = len(ref_audio) / TARGET_SR
    if ref_duration_sec < 0.1:
         raise HTTPException(status_code=400, detail="Reference audio too short < 0.1s")

    # Match server max_sequence_length (overrideable via env var for the combined launcher).
    try:
        MAX_SEQUENCE_LENGTH = int(os.environ.get("F5_MAX_SEQUENCE_LENGTH", "2048"))
    except ValueError:
        MAX_SEQUENCE_LENGTH = 2048
    
    gen_text = request.text
    ref_text = request.ref_text
    speed_factor = request.speed
    
    # Ensure reference text ends with space if last char is ASCII
    if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    if request.gen_len is not None:
        # Fixed duration mode - Treats text as single chunk
        gen_text_batches = [gen_text]
        chunk_durations = [request.gen_len]
        num_chunks = 1
    else:
        # Standard mode - Dynamic chunking and duration
        chars_per_sec_ref = len(ref_text.encode("utf-8")) / ref_duration_sec
        max_gen_duration_sec = MAX_SEQUENCE_LENGTH * 256 / TARGET_SR - ref_duration_sec
        
        if max_gen_duration_sec <= 0:
             raise HTTPException(status_code=400, detail="Reference audio too long for max sequence length")

        estimated_max_chars = max(10, int(chars_per_sec_ref * max_gen_duration_sec * 0.8 * speed_factor))
        
        gen_text_batches = chunk_text(gen_text, max_chars=estimated_max_chars)
        num_chunks = len(gen_text_batches)
        
        if num_chunks == 0:
             raise HTTPException(status_code=400, detail="No chunks generated")
        
        if num_chunks > MAX_CHUNKS:
             raise HTTPException(status_code=400, detail=f"Text too long, resulted in {num_chunks} chunks (max {MAX_CHUNKS})")
        
        chunk_durations = [] # To be filled later


    # Truncate Ref Audio Logic
    hop_length = 256
    ref_audio_len_frames = ref_audio.shape[-1] // hop_length + 1
    max_ref_frames = int(MAX_SEQUENCE_LENGTH * 0.6)
    
    if ref_audio_len_frames > max_ref_frames:
        ref_audio_len_frames = max_ref_frames
        ref_audio = ref_audio[:ref_audio_len_frames * hop_length]
        # Approximate text truncation
        original_ref_text_len = len(ref_text)
        ref_text = ref_text[:int(original_ref_text_len * (max_ref_frames / (ref_audio.shape[-1] // hop_length + 1)))]
        if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
             ref_text += " "

    # Prepare chunks
    futures_data = [] # To send sequentially or batch? 
    # The REQ socket is strict request-reply. We can't pipeline efficiently with a SINGLE REQ socket if we want to stream parallel 
    # UNLESS the server supports batching multiple chunks in one request OR we handle one by one.
    # The original was parallel futures.
    # Approach: Send separate requests to server? 
    # With REQ/ROUTER, REQ must send, then recv. It blocks.
    # If we want parallel processing on server side, we need multiple connections or DEALER?
    # BUT wait, the requirements was "FastAPI separate process, JAX separate process".
    # If we iterate here:
    # for chunk in chunks: await socket.send(); await socket.recv() -> SERIAL processing on inference side (mostly). Note: JAX is batched.
    # If we send 5 requests one by one, they enter the server queue. The server batch manager picks them up.
    # BUT since REQ is lock-step, we wait for 1 to finish before sending 2. That defeats batching if we only have ONE connection.
    # 
    # Solution 1: Send ALL chunks in ONE request.
    # Solution 2: Use many connections (expensive/complex).
    #
    # Best for batching: Send a list of items in one ZMQ message?
    # BUT the Server BatchManager treats each "Input" as an item.
    # The Server receives "one message" -> puts in queue.
    # If we change server to accept "List of Requests", it complicates the uniqueness/identity mapping.
    #
    # ACTUALLY, simpler:
    # Just send each chunk as a separate async Task, but we need a POOL of connections or DEALER socket to be async?
    # zmq.asyncio REQ socket: "send, then recv". You cannot send twice before recv.
    #
    # Let's switch to DEALER for the API side? 
    # If we use DEALER, we can fire off N requests. We need to track which reply matches which request (via internal ID or order?).
    # Order is preserved in ZMQ pairs usually if single path.
    # Or, simpler:
    # Just loop and send one by one?
    # If we send one by one:
    #   req1 -> server. Server puts in queue. 
    #   server worker picks up... waits for batch...
    #   req1 waits...
    #   req2 cannot be sent because req1 is waiting for reply!
    #
    # Correct Architecture for Batching from ONE client:
    # The API Server should probably send the WHOLE BATCH of chunks as ONE technical request to the Inference Server?
    # And the Inference Server splits it?
    # OR
    # The API Server uses a DEALER socket.
    # Let's update `f5_api.py` to use DEALER.
    # With DEALER, we can `await socket.send` N times. Then `await socket.recv` N times.
    # We just need to start N tasks.
    
    # Let's try DEALER.
    
    batched_text_list_combined = []
    batched_duration_frames = []
    
    for single_gen_text in gen_text_batches:
        text_combined = ref_text + single_gen_text
        batched_text_list_combined.append(text_combined)

        if request.gen_len is not None:
             # Use fixed duration provided
             estimated_gen_frames = request.gen_len
        else:
            ref_text_byte_len = len(ref_text.encode('utf-8'))
            gen_text_byte_len = len(single_gen_text.encode('utf-8'))

            if ref_text_byte_len > 0:
                 estimated_gen_frames = int(ref_audio_len_frames / ref_text_byte_len * gen_text_byte_len / speed_factor)
            else:
                 avg_chars_per_sec = 5 * speed_factor
                 estimated_gen_frames = int(gen_text_byte_len * (TARGET_SR / hop_length) / avg_chars_per_sec) if avg_chars_per_sec > 0 else 50


        estimated_gen_frames = max(0, estimated_gen_frames)
        duration_frames = ref_audio_len_frames + estimated_gen_frames
        
        if duration_frames > MAX_SEQUENCE_LENGTH:
             raise HTTPException(status_code=400, detail=f"Audio too long. Total frames: {duration_frames} > Max: {MAX_SEQUENCE_LENGTH}")

        duration_frames = min(MAX_SEQUENCE_LENGTH, duration_frames)
        duration_frames = max(ref_audio_len_frames + 1, duration_frames)

        batched_duration_frames.append(duration_frames)

    # Sending
    tasks = []
    for i in range(num_chunks):
        payload = {
            "text": batched_text_list_combined[i],
            "ref_audio": ref_audio,
            "duration_frames": batched_duration_frames[i],
            "ref_text": ref_text
        }
        # Create a unique ID for this chunk request if needed, or rely on order? 
        # With asyncio DEALER, we can map tasks?
        # Actually with DEALER, send is fire-and-forget (roughly). 
        # But we need to match responses.
        #
        # Let's assume strict FIFO order of replies from the server? 
        # The servers BatchManager processes batches. Returns results.
        # It's possible the server processes out of order if we had multiple workers (we have 1).
        # So order should be preserved.
        tasks.append(send_request(payload))
    
    t1 = time.time()
    results = await asyncio.gather(*tasks)
    t2 = time.time()
    
    # results is list of numpy arrays
    final_audio_segments = []
    ref_len_samples = ref_audio_len_frames * hop_length

    for i, audio_seg in enumerate(results):
        current_duration_frames = batched_duration_frames[i]
        current_duration_samples = current_duration_frames * hop_length
        # Crop
        generated_part = audio_seg[ref_len_samples:current_duration_samples]
        final_audio_segments.append(generated_part)

    final_audio = np.concatenate(final_audio_segments) if final_audio_segments else np.array([], dtype=np.float32)

    buffer_io = io.BytesIO()
    sf.write(buffer_io, final_audio, TARGET_SR, format='WAV')
    audio_b64 = base64.b64encode(buffer_io.getvalue()).decode('utf-8')

    t3 = time.time()
    if _timing_enabled():
        print(
            f"Preprocessing: {t1-t0:.4f}s, Inference: {t2-t1:.4f}s, "
            f"Postprocessing: {t3-t2:.4f}s, Total: {t3-t0:.4f}s"
        )

    return GenerateResponse(audio_base64=audio_b64)


async def send_request(payload):
    # We need to lock the socket if we want to ensure message integrity? 
    # zmq.asyncio socket sending is thread-safe?
    # Ideally we'd use a pool or a new socket per request if we want massive concurrency?
    # Simpler: Create a new DEALER/REQ for each chunk? No that's overhead.
    # 
    # Re-evaluating: 'f5_api.py' is a SERVER. Multiple users can hit it.
    # If we share ONE global 'zmq_socket' (DEALER) for all incoming FastAPI requests -> 
    # User A sends 5 chunks. User B sends 5 chunks.
    # Messages A1..A5, B1..B5 interleaved.
    # Recv loop gets results. How do we know which result belongs to User A or B?
    #
    # We NEED a request ID content in the payload, and a global dict of pending futures.
    #
    # Plan B (Simpler/Robust):
    # Use a fresh REQ socket per HTTP request? 
    # Overhead of TCP connect to localhost is small. 
    # ZMQ sockets are fast.
    # 
    # Let's try: Context is global. Socket is created PER request or PER chunk?
    # Per HTTP request -> Socket (DEALER) -> Send N chunks -> Recv N chunks -> Close.
    # This ensures isolation between User A and User B.
    #
    # Code updated to do local socket.
    
    local_socket = zmq_context.socket(zmq.DEALER)
    # Identity must be unique otherwise server might get confused if we reuse check
    import uuid
    local_socket.identity = str(uuid.uuid4()).encode('ascii') 
    local_socket.connect(SERVER_ADDRESS)
    
    # Send
    serialized = pickle.dumps(payload)
    await local_socket.send(serialized)
    
    # Recv
    # DEALER gets: [empty?, data] or just data? 
    # ROUTER sends: [identity, empty, data] -> DEALER strips identity. 
    # DEALER usually receives: [empty, data] or [data] depending on sender.
    # My server sends: socket.send_multipart([identity, b'', serialized_result])
    # So DEALER should see [b'', serialized_result]
    parts = await local_socket.recv_multipart()
    if len(parts) > 1 and parts[0] == b'':
        resp_data = parts[1]
    else:
        resp_data = parts[0]
        
    local_socket.close()
    
    resp = pickle.loads(resp_data)
    if resp.get("status") != "ok":
        raise Exception(resp.get("message", "Unknown error"))
    
    return resp["audio"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
