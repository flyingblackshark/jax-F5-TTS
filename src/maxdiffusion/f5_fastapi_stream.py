from typing import Callable, List, Union, Sequence, Tuple, AsyncGenerator
from absl import app
from contextlib import ExitStack
import functools
import jax.experimental
import jax.experimental.compilation_cache.compilation_cache
import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec as P
import jax.numpy as jnp
import flax
from maxdiffusion import pyconfig, max_logging
from maxdiffusion.models.f5.transformers.transformer_f5_flax import F5TextEmbedding, F5Transformer2DModel
from maxdiffusion.max_utils import (
    device_put_replicated,
    get_memory_allocations,
    create_device_mesh,
    get_flash_block_sizes,
    get_precision,
    setup_initial_state,
)
import time
from maxdiffusion.models.modeling_flax_pytorch_utils import convert_f5_state_dict_to_flax
import librosa
import jax.experimental.compilation_cache
from jax_vocos import load_model as load_vocos_model
from maxdiffusion.utils.mel_util import get_mel
from maxdiffusion.utils.pinyin_utils import get_tokenizer, chunk_text, convert_char_to_pinyin, list_str_to_idx
from maxdiffusion.utils.seq_utils import lens_to_mask
import asyncio
import io
import base64
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json

# --- Configuration & Constants ---
jax.experimental.compilation_cache.compilation_cache.set_cache_dir("./jax_cache")
cfg_strength = 2.0
TARGET_SR = 24000
MAX_DURATION_SECS = 40
MAX_INFERENCE_STEPS = 100
DEFAULT_REF_TEXT = "and there are so many things about humankind that is bad and evil. I strongly believe that love is one of the only things we have in this world."

# --- Global Variables for Model State ---
global_config = None
global_mesh = None
global_transformer = None
global_transformer_state = None
global_transformer_state_shardings = None
global_text_encoder = None
global_text_encoder_params = None
global_jitted_text_encode = None
global_vocos_model = None
global_vocos_params = None
global_jitted_vocos_apply = None
global_vocab_char_map = None
global_vocab_size = None
global_p_run_inference = None
global_data_sharding = None
global_max_sequence_length = None
jitted_get_mel = None

# --- FastAPI Models ---
class AudioGenerationRequest(BaseModel):
    ref_text: str
    gen_text: str
    ref_audio_base64: str  # Base64 encoded audio data
    num_inference_steps: int = 50
    guidance_scale: float = 2.0
    speed_factor: float = 1.0
    use_sway_sampling: bool = False
    sequence_length: int = 1024  # 512, 1024, 2048, 4096
    chunk_size: int = 1024  # Audio chunk size for streaming

class StreamChunk(BaseModel):
    audio_chunk: str  # Base64 encoded audio chunk
    is_final: bool = False
    chunk_index: int = 0
    total_chunks: int = 0

# --- Core Diffusion Loop Logic (Same as original) ---
def loop_body(
    step,
    args,
    transformer,
    cond,
    decoder_segment_ids,
    text_embed_cond,
    text_embed_uncond,
):
    latents, state, c_ts, p_ts = args
    latents_dtype = latents.dtype
    t_curr = c_ts[step]
    t_prev = p_ts[step]
    t_vec = jnp.full((latents.shape[0],), t_curr, dtype=latents.dtype)

    # Conditional prediction
    pred = transformer.apply(
        {"params": state.params},
        x=latents,
        cond=cond,
        decoder_segment_ids=decoder_segment_ids,
        text_embed=text_embed_cond,
        timestep=t_vec,
    )

    # Unconditional prediction
    null_pred = transformer.apply(
        {"params": state.params},
        x=latents,
        cond=jnp.zeros_like(cond),
        decoder_segment_ids=decoder_segment_ids,
        text_embed=text_embed_uncond,
        timestep=t_vec,
    )

    # Classifier-Free Guidance
    guidance_scale = cfg_strength
    pred = null_pred + guidance_scale * (pred - null_pred)

    # DDIM-like step
    latents = latents + (t_prev - t_curr) * pred
    latents = jnp.array(latents, dtype=latents_dtype)

    return latents, state, c_ts, p_ts

def run_inference(
    states, latents, cond, decoder_segment_ids, text_embed_cond, text_embed_uncond, c_ts, p_ts, transformer, config, mesh
):
    transformer_state = states

    loop_body_p = functools.partial(
        loop_body,
        transformer=transformer,
        cond=cond,
        decoder_segment_ids=decoder_segment_ids,
        text_embed_cond=text_embed_cond,
        text_embed_uncond=text_embed_uncond,
    )

    latents_final, _, _, _ = jax.lax.fori_loop(0, len(c_ts), loop_body_p, (latents, transformer_state, c_ts, p_ts))
    return latents_final

# --- Streaming Audio Generation Function ---
async def generate_audio_stream(
    ref_text: str,
    gen_text: str,
    ref_audio: np.ndarray,
    num_inference_steps: int = 50,
    guidance_scale: float = 2.0,
    speed_factor: float = 1.0,
    use_sway_sampling: bool = False,
    sequence_length: int = 1024,
    chunk_size: int = 1024
) -> AsyncGenerator[StreamChunk, None]:
    """
    Streaming version of audio generation that yields audio chunks as they are generated.
    """
    global cfg_strength
    cfg_strength = guidance_scale
    
    # Override global max sequence length for this request
    current_max_length = sequence_length
    
    t_start_total = time.time()
    max_logging.log(f"Starting streaming audio generation... Steps: {num_inference_steps}, CFG: {guidance_scale}, Speed: {speed_factor}, Seq Length: {sequence_length}")

    # --- Input Validation ---
    if not ref_text:
        ref_text = DEFAULT_REF_TEXT
    if not gen_text:
        raise ValueError("Generation text cannot be empty.")
    if ref_audio is None or ref_audio.size == 0:
        raise ValueError("Reference audio is required.")

    # --- Preprocessing ---
    t_start_preprocess = time.time()
    max_logging.log("Preprocessing text and audio...")

    # Ensure reference text ends with space if last char is ASCII
    if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    # Calculate reference audio duration
    ref_duration_sec = len(ref_audio) / TARGET_SR
    if ref_duration_sec < 0.1:
        raise ValueError("Reference audio is too short (must be at least 0.1 seconds).")

    # For single TPU v6e-1, use batch_size = 1
    batch_size = 1
    
    # Estimate character count and chunk text
    chars_per_sec_ref = len(ref_text.encode("utf-8")) / ref_duration_sec
    max_gen_duration_sec = MAX_DURATION_SECS - ref_duration_sec
    if max_gen_duration_sec <= 0:
        raise ValueError(f"Reference audio duration ({ref_duration_sec:.1f}s) exceeds max allowed duration ({MAX_DURATION_SECS}s).")

    estimated_max_chars = max(10, int(chars_per_sec_ref * max_gen_duration_sec * 0.8 * speed_factor))
    gen_text_batches = chunk_text(gen_text, max_chars=estimated_max_chars)
    num_chunks = len(gen_text_batches)
    
    if num_chunks == 0:
        raise ValueError("Text processing resulted in zero valid chunks.")
    
    max_logging.log(f"Split generation text into {num_chunks} chunks.")

    # Process each chunk and stream results
    hop_length = 256
    ref_audio_len_frames = ref_audio.shape[-1] // hop_length + 1
    
    # Limit reference audio to avoid exceeding sequence length
    max_ref_frames = int(current_max_length * 0.6)
    if ref_audio_len_frames > max_ref_frames:
        ref_audio_len_frames = max_ref_frames
        ref_audio = ref_audio[:ref_audio_len_frames * hop_length]
        original_ref_text_len = len(ref_text)
        ref_text = ref_text[:int(original_ref_text_len * (max_ref_frames / (ref_audio.shape[-1] // hop_length + 1)))]
        if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
            ref_text += " "

    # Prepare reference audio condition
    ref_audio_padded = np.pad(ref_audio, (0, max(0, current_max_length * hop_length + hop_length - ref_audio.shape[0])))
    ref_audio_padded = ref_audio_padded[np.newaxis, :]
    cond = jitted_get_mel(ref_audio_padded)
    cond_pad_len = current_max_length - cond.shape[1]
    if cond_pad_len > 0:
        cond = np.pad(cond, ((0,0), (0, cond_pad_len), (0,0)))
    elif cond_pad_len < 0:
        cond = cond[:, :current_max_length, :]
    
    # Broadcast condition to batch size
    cond = np.repeat(cond, batch_size, axis=0)
    
    t_end_preprocess = time.time()
    max_logging.log(f"Preprocessing finished in {t_end_preprocess - t_start_preprocess:.2f}s.")

    # Process chunks and stream audio
    total_audio_chunks = 0
    for i, single_gen_text in enumerate(gen_text_batches):
        chunk_start_time = time.time()
        
        # Prepare text for this chunk
        text_combined = ref_text + single_gen_text
        
        # Estimate duration for this chunk
        ref_text_byte_len = len(ref_text.encode('utf-8'))
        gen_text_byte_len = len(single_gen_text.encode('utf-8'))
        
        if ref_text_byte_len > 0:
            estimated_gen_frames = int(ref_audio_len_frames / ref_text_byte_len * gen_text_byte_len / speed_factor)
        else:
            avg_chars_per_sec = 5 * speed_factor
            estimated_gen_frames = int(gen_text_byte_len * (TARGET_SR / hop_length) / avg_chars_per_sec) if avg_chars_per_sec > 0 else 50
        
        estimated_gen_frames = max(0, estimated_gen_frames)
        duration_frames = ref_audio_len_frames + estimated_gen_frames
        duration_frames = min(current_max_length, duration_frames)
        duration_frames = max(ref_audio_len_frames + 1, duration_frames)
        
        # Convert text to pinyin and tokenize
        final_text_list_pinyin = convert_char_to_pinyin([text_combined])
        text_ids = list_str_to_idx(final_text_list_pinyin, global_vocab_char_map, max_length=current_max_length)
        
        # Pad to batch size
        text_ids = np.pad(text_ids, ((0, batch_size - text_ids.shape[0]), (0, 0)), constant_values=0)
        
        # Prepare masks and segment IDs
        ref_len_frames_arr = np.array([ref_audio_len_frames] * batch_size, dtype=np.int32)
        duration_frames_arr = np.array([duration_frames] * batch_size, dtype=np.int32)
        text_lens = np.minimum((text_ids != 0).sum(axis=-1), current_max_length)
        
        effective_min_len = np.maximum(text_lens, ref_len_frames_arr) + 1
        duration_final = np.maximum(effective_min_len, duration_frames_arr)
        duration_final = np.minimum(duration_final, current_max_length)
        
        cond_mask = lens_to_mask(ref_len_frames_arr, length=current_max_length)
        decoder_mask = lens_to_mask(duration_final, length=current_max_length)
        
        text_decoder_segment_ids = (text_ids != 0).astype(np.int32)
        decoder_segment_ids = decoder_mask.astype(np.int32)
        
        step_cond = np.where(cond_mask[..., np.newaxis], cond, np.zeros_like(cond))
        
        # Shard data
        step_cond = jax.device_put(step_cond, global_data_sharding)
        text_ids = jax.device_put(text_ids, global_data_sharding)
        decoder_segment_ids = jax.device_put(decoder_segment_ids, global_data_sharding)
        text_decoder_segment_ids = jax.device_put(text_decoder_segment_ids, global_data_sharding)
        cond_mask_sharded = jax.device_put(cond_mask, global_data_sharding)
        
        # Generate text embeddings
        rng_embed = jax.random.key(global_config.seed + 1 + i)
        rngs_embed = {'params': rng_embed, 'dropout': rng_embed}
        
        text_embed_cond = global_jitted_text_encode_func(
            {"params": global_text_encoder_params},
            text_ids,
            text_decoder_segment_ids,
            rngs_embed
        )
        
        text_embed_uncond = global_jitted_text_encode_func(
            {"params": global_text_encoder_params},
            np.zeros_like(text_ids),
            text_decoder_segment_ids,
            rngs_embed
        )
        
        # Diffusion sampling
        latents_shape = (batch_size, current_max_length, 100)
        latents_rng = jax.random.key(global_config.seed + 2 + i)
        latents = jax.random.normal(latents_rng, latents_shape, dtype=jnp.float32)
        latents = jax.device_put(latents, global_data_sharding)
        
        # Timestep calculation
        t_start = 0.0
        timesteps = np.linspace(t_start, 1.0, num_inference_steps + 1).astype(np.float32)
        
        if use_sway_sampling and hasattr(global_config, 'sway_sampling_coef') and global_config.sway_sampling_coef:
            sway_coef = global_config.sway_sampling_coef
            timesteps = timesteps + sway_coef * (np.cos(np.pi / 2 * timesteps) - 1 + timesteps)
            timesteps = np.clip(timesteps, 0.0, 1.0)
        
        c_ts = timesteps[:-1]
        p_ts = timesteps[1:]
        
        # Run inference
        y_final_latents = global_p_run_inference_func(
            global_transformer_state,
            latents,
            step_cond,
            decoder_segment_ids,
            text_embed_cond,
            text_embed_uncond,
            c_ts,
            p_ts
        )
        
        y_final_latents.block_until_ready()
        
        # Apply vocoder
        out_latents = jnp.where(cond_mask_sharded[..., jnp.newaxis], cond, y_final_latents)
        
        vocoder_rng = jax.random.key(global_config.seed + 3 + i)
        rngs_vocoder = {'params': vocoder_rng, 'dropout': vocoder_rng}
        
        audio_out_jax = global_jitted_vocos_apply_func(
            {"params": global_vocos_params},
            out_latents,
            rngs_vocoder
        )
        audio_out_jax.block_until_ready()
        
        # Transfer to CPU and extract generated part
        audio_out_cpu = np.asarray(audio_out_jax[0])  # Take first batch item
        ref_len_samples = ref_audio_len_frames * hop_length
        current_duration_samples = duration_frames * hop_length
        
        generated_part = audio_out_cpu[ref_len_samples:current_duration_samples]
        
        # Split generated audio into chunks for streaming
        num_audio_chunks = max(1, len(generated_part) // chunk_size)
        
        for chunk_idx in range(num_audio_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(generated_part))
            audio_chunk = generated_part[start_idx:end_idx]
            
            # Convert to bytes and encode as base64
            audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            is_final = (i == len(gen_text_batches) - 1) and (chunk_idx == num_audio_chunks - 1)
            
            chunk_data = StreamChunk(
                audio_chunk=audio_base64,
                is_final=is_final,
                chunk_index=total_audio_chunks,
                total_chunks=sum(max(1, len(chunk) // chunk_size) for chunk in gen_text_batches)
            )
            
            total_audio_chunks += 1
            yield chunk_data
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)
        
        chunk_end_time = time.time()
        max_logging.log(f"Chunk {i+1}/{len(gen_text_batches)} processed in {chunk_end_time - chunk_start_time:.2f}s")
    
    t_end_total = time.time()
    max_logging.log(f"Total streaming generation completed in {t_end_total - t_start_total:.2f}s")

# --- Setup Function (Same as original but adapted for single TPU) ---
def setup_models_and_state(config):
    """
    Initialize models and state for single TPU v6e-1.
    """
    global global_config, global_mesh, global_transformer, global_transformer_state
    global global_transformer_state_shardings, global_text_encoder, global_text_encoder_params
    global global_jitted_text_encode_func, global_vocos_model, global_vocos_params
    global global_jitted_vocos_apply_func, global_vocab_char_map, global_vocab_size
    global global_p_run_inference_func, global_data_sharding, global_max_sequence_length
    global jitted_get_mel

    t_start_setup = time.time()
    max_logging.log("Starting one-time setup for single TPU v6e-1...")
    global_config = config

    flash_block_sizes = get_flash_block_sizes(config)
    global_max_sequence_length = config.max_sequence_length
    max_logging.log(f"Model configured for max sequence length: {global_max_sequence_length}")

    rng = jax.random.key(config.seed)
    devices_array = create_device_mesh(config)
    global_mesh = Mesh(devices_array, config.mesh_axes)
    mesh = global_mesh

    if not config.mesh_axes:
        raise ValueError("config.mesh_axes must be defined (e.g., ['data'])")
    data_axis_name = config.mesh_axes[0]
    max_logging.log(f"Using mesh axes: {config.mesh_axes} (Data axis: '{data_axis_name}')")

    # Define sharding specs for single TPU
    sharding_spec_batch_only = P(data_axis_name)
    sharding_spec_batch_seq = P(data_axis_name, None)
    sharding_spec_batch_seq_dim = P(data_axis_name, None, None)
    sharding_spec_get_mel_input = P(None, data_axis_name)
    sharding_spec_get_mel_output = P(None, data_axis_name, None)

    # JIT compile get_mel with sharding
    max_logging.log("Compiling get_mel with sharding...")
    get_mel_in_shardings = (jax.sharding.NamedSharding(mesh, sharding_spec_get_mel_input),)
    get_mel_out_shardings = None

    jitted_get_mel = jax.jit(
        get_mel,
        static_argnums=(1, 2, 3, 4, 5, 6, 8),
        in_shardings=get_mel_in_shardings,
        out_shardings=get_mel_out_shardings
    )

    # Warmup get_mel
    try:
        hop_length = 256
        dummy_audio_len = global_max_sequence_length * hop_length + hop_length
        compile_batch_size = 1
        dummy_audio_shape = (compile_batch_size, dummy_audio_len)
        dummy_audio = jnp.zeros(dummy_audio_shape, dtype=jnp.float32)
        dummy_audio_sharded = jax.device_put(dummy_audio, get_mel_in_shardings[0])
        
        max_logging.log(f"Warming up jitted_get_mel with dummy shape {dummy_audio_shape}...")
        _ = jitted_get_mel(dummy_audio_sharded).block_until_ready()
        max_logging.log("jitted_get_mel successfully compiled and warmed up.")
    except Exception as e:
        max_logging.error(f"Failed to pre-compile/warmup jitted_get_mel: {e}", exc_info=True)
        raise

    # Load Transformer
    max_logging.log("Loading F5 Transformer model...")
    global_transformer = F5Transformer2DModel(
        text_dim=config.text_dim,
        mel_dim=config.mel_dim,
        dim=config.latent_dim,
        head_dim=config.head_dim,
        num_depth=config.num_depth,
        num_heads=config.num_heads,
        mesh=mesh,
        attention_kernel=config.attention,
        flash_block_sizes=flash_block_sizes,
        dtype=config.activations_dtype,
        weights_dtype=config.weights_dtype,
        precision=get_precision(config),
    )
    transformer = global_transformer

    # Load weights
    transformer_params, text_encoder_params_loaded = convert_f5_state_dict_to_flax(
        config.pretrained_model_name_or_path, use_ema=config.use_ema
    )
    global_text_encoder_params = flax.core.frozen_dict.FrozenDict(text_encoder_params_loaded)

    weights_init_fn = functools.partial(
        transformer.init_weights, 
        rngs=rng, 
        max_sequence_length=config.max_sequence_length, 
        eval_only=False
    )
    global_transformer_state, global_transformer_state_shardings = setup_initial_state(
        model=transformer,
        tx=None,
        config=config,
        mesh=mesh,
        weights_init_fn=weights_init_fn,
        model_params=None,
        training=False,
    )
    global_transformer_state = global_transformer_state.replace(params=transformer_params)
    global_transformer_state = jax.device_put(global_transformer_state, global_transformer_state_shardings)

    # Load Text Encoder
    max_logging.log("Loading Text Encoder model...")
    global_vocab_char_map, global_vocab_size = get_tokenizer(config.vocab_name_or_path, "custom")

    global_text_encoder = F5TextEmbedding(
        text_num_embeds=2545,
        text_dim=512,
        conv_layers=4,
        dtype=jnp.float32
    )
    text_encoder = global_text_encoder

    global_text_encoder_params = jax.device_put(global_text_encoder_params, None)
    max_logging.log("Text encoder params replicated on devices.")

    text_encode_in_shardings = (
        None,
        jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq),
        jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq),
        None
    )
    text_encode_out_shardings = jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq_dim)

    def wrap_text_encoder_apply(params, text_ids, text_decoder_segment_ids, rngs):
        return text_encoder.apply(params, text_ids, text_decoder_segment_ids, rngs=rngs)

    global_jitted_text_encode_func = jax.jit(
        wrap_text_encoder_apply,
        in_shardings=text_encode_in_shardings,
        out_shardings=text_encode_out_shardings,
        static_argnums=()
    )

    max_logging.log("Text Encoder JIT compiled.")

    # Load Vocoder
    max_logging.log("Loading Vocoder model...")
    global_vocos_model, vocos_params_loaded = load_vocos_model(config.vocoder_model_path)
    vocos_model = global_vocos_model
    global_vocos_params = flax.core.frozen_dict.FrozenDict(vocos_params_loaded)

    global_vocos_params = jax.device_put(global_vocos_params, None)
    max_logging.log("Vocoder params replicated on devices.")

    vocos_apply_in_shardings = (
        None,
        jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq_dim),
        None,
    )
    vocos_apply_out_shardings = jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq)

    def wrap_vocos_apply(params, x, rngs):
        return vocos_model.apply(params, x, rngs=rngs)

    global_jitted_vocos_apply_func = jax.jit(
        wrap_vocos_apply,
        in_shardings=vocos_apply_in_shardings,
        out_shardings=vocos_apply_out_shardings,
        static_argnums=()
    )
    max_logging.log("Vocoder JIT compiled.")

    # Compile Inference Loop
    max_logging.log("Compiling main inference loop...")
    global_data_sharding = jax.sharding.NamedSharding(mesh, P(config.data_sharding[0]))

    latents_sharding = global_data_sharding
    cond_sharding = global_data_sharding
    decoder_segment_ids_sharding = global_data_sharding
    text_embed_sharding = global_data_sharding
    ts_sharding = None

    partial_run_inference = functools.partial(
        run_inference,
        transformer=transformer,
        config=config,
        mesh=mesh,
    )

    in_shardings_inf = (
        global_transformer_state_shardings,
        latents_sharding,
        cond_sharding,
        decoder_segment_ids_sharding,
        text_embed_sharding,
        text_embed_sharding,
        ts_sharding,
        ts_sharding
    )
    out_shardings_inf = latents_sharding

    try:
        global_p_run_inference_func = jax.jit(
            partial_run_inference,
            static_argnums=(),
            in_shardings=in_shardings_inf,
            out_shardings=out_shardings_inf,
        )
        max_logging.log("Inference loop JIT compiled.")
    except Exception as e:
        max_logging.error(f"Failed to pre-compile inference loop: {e}")
        raise

    t_end_setup = time.time()
    max_logging.log(f"One-time setup completed in {t_end_setup - t_start_setup:.2f}s.")
    get_memory_allocations()

# --- FastAPI Application ---
app = FastAPI(title="F5-TTS Streaming API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    try:
        # Initialize pyconfig with default arguments
        pyconfig.initialize([])
        config = pyconfig.config
        
        # Override config for single TPU v6e-1
        config.per_device_batch_size = 1
        config.mesh_axes = ['data']
        
        setup_models_and_state(config)
        max_logging.log("FastAPI server startup completed successfully.")
    except Exception as e:
        max_logging.error(f"Fatal error during startup: {e}", exc_info=True)
        raise

@app.post("/generate_stream")
async def generate_audio_endpoint(request: AudioGenerationRequest):
    """Generate audio with streaming response."""
    try:
        # Decode base64 audio
        ref_audio_bytes = base64.b64decode(request.ref_audio_base64)
        ref_audio = np.frombuffer(ref_audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
        # Validate sequence length
        valid_lengths = [512, 1024, 2048, 4096]
        if request.sequence_length not in valid_lengths:
            raise HTTPException(status_code=400, f"Invalid sequence_length. Must be one of {valid_lengths}")
        
        async def stream_generator():
            try:
                async for chunk in generate_audio_stream(
                    ref_text=request.ref_text,
                    gen_text=request.gen_text,
                    ref_audio=ref_audio,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    speed_factor=request.speed_factor,
                    use_sway_sampling=request.use_sway_sampling,
                    sequence_length=request.sequence_length,
                    chunk_size=request.chunk_size
                ):
                    yield f"data: {chunk.json()}\n\n"
                
                # Send final message
                yield "data: {\"is_final\": true}\n\n"
            except Exception as e:
                error_msg = {"error": str(e), "is_final": True}
                yield f"data: {json.dumps(error_msg)}\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    
    except Exception as e:
        max_logging.error(f"Error in generate_audio_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": global_transformer is not None}

@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "max_sequence_length": global_max_sequence_length,
        "supported_sequence_lengths": [512, 1024, 2048, 4096],
        "target_sample_rate": TARGET_SR,
        "max_duration_seconds": MAX_DURATION_SECS
    }

def main(argv: Sequence[str]) -> None:
    """Main function to start the FastAPI server."""
    max_logging.log("Starting F5-TTS FastAPI Streaming Server...")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    import sys
    main(sys.argv)