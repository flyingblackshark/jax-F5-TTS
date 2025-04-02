import gradio as gr # Import Gradio
from typing import Callable, List, Union, Sequence, Tuple
from absl import app
from contextlib import ExitStack
import functools
import jax.experimental
import jax.experimental.compilation_cache.compilation_cache
import jax.experimental.ode
import numpy as np
import jax
from jax.sharding import Mesh, PositionalSharding, PartitionSpec as P
import jax.numpy as jnp
import flax.linen as nn
from chex import Array
from einops import rearrange
from flax.linen import partitioning as nn_partitioning
import flax
import re
from pypinyin import lazy_pinyin, Style
import jieba
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
import os
from importlib.resources import files
import librosa
import audax.core.functional
import jax.experimental.compilation_cache
from jax_vocos import load_model as load_vocos_model # Renamed to avoid conflict
import soundfile as sf
import io

# --- Configuration & Constants ---
jax.experimental.compilation_cache.compilation_cache.set_cache_dir("./jax_cache")
cfg_strength = 2.0 # Made this a variable, potentially could be a Gradio slider
TARGET_SR = 24000
MAX_DURATION_SECS = 40 # Maximum duration allowed for reference + generation combined (adjust as needed)
MAX_INFERENCE_STEPS = 100 # Default inference steps, could be Gradio input
DEFAULT_REF_TEXT = "and there are so many things about humankind that is bad and evil. I strongly believe that love is one of the only things we have in this world."
# === Add Bucket Constants ===
BUCKET_SIZES = sorted([4, 8, 16, 32, 64])
MAX_CHUNKS = BUCKET_SIZES[-1]
# ==========================

# --- JAX/Model Setup (Global Scope for Gradio) ---
# These will be initialized once when the script starts
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
global_max_sequence_length = None # Will be set during setup
#global_batch_size = None # Will be set during setup

# --- Utility Functions (Mostly unchanged, slight modifications) ---

def dynamic_range_compression_jax(x, C=1, clip_val=1e-7):
    return jnp.log(jnp.clip(x, min=clip_val) * C)

def get_mel(y, n_mels=100, n_fft=1024, win_size=1024, hop_length=256, fmin=0, fmax=None, clip_val=1e-7, sampling_rate=TARGET_SR):
    # Ensure input is JAX array
    y = jnp.asarray(y)
    # Ensure it's mono
    if y.ndim > 1 and y.shape[0] > 1:
        y = jnp.mean(y, axis=0)
    elif y.ndim > 1:
        y = jnp.squeeze(y, axis=0)

    window = jnp.hanning(win_size)
    spec_func = functools.partial(audax.core.functional.spectrogram, pad=0, window=window, n_fft=n_fft,
                    hop_length=hop_length, win_length=win_size, power=1.,
                    normalized=False, center=True, onesided=True)
    fb = audax.core.functional.melscale_fbanks(n_freqs=(n_fft // 2) + 1, n_mels=n_mels,
                        sample_rate=sampling_rate, f_min=fmin, f_max=fmax)
    mel_spec_func = functools.partial(audax.core.functional.apply_melscale, melscale_filterbank=fb)
    spec = spec_func(y)
    spec = mel_spec_func(spec)
    spec = dynamic_range_compression_jax(spec, clip_val=clip_val)
    return spec

# JIT get_mel for performance
#jitted_get_mel = jax.jit(get_mel, static_argnums=(1, 2, 3, 4, 5, 6, 8))

def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


def get_tokenizer(dataset_name, tokenizer: str = "custom"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in ["pinyin", "char"]:
        tokenizer_path = os.path.join(files("f5_tts").joinpath("../../data"), f"{dataset_name}_{tokenizer}/vocab.txt")
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)
        assert vocab_char_map[" "] == 0, "make sure space is of idx 0 in vocab.txt, cuz 0 is used for unknown char"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size

def list_str_to_idx(
    text: list[list[str]], # Expects list of lists of chars/pinyin
    vocab_char_map: dict[str, int],
    max_length: int,
    padding_value=0, # Use 0 for padding index (which maps to space or unknown)
):
    outs = []
    #unk_idx = vocab_char_map.get('<unk>', vocab_char_map.get(' ', 0)) # Use space if <unk> not present

    for t in text:
        # Map characters/pinyin, using unk_idx for unknown ones
        list_idx_tensors = [vocab_char_map.get(c, 0) for c in t]
        text_ids = np.asarray(list_idx_tensors, dtype=np.int32)

        # Add 1 to all indices (making padding 1, original indices shifted)
        text_ids = text_ids + 1 # Let's reconsider this, maybe padding with 0 is better if space is 0

        # Pad sequence
        pad_len = max_length - text_ids.shape[-1]
        if pad_len < 0:
            print(f"Warning: Truncating text sequence from {text_ids.shape[-1]} to {max_length}")
            text_ids = text_ids[:max_length]
            pad_len = 0

        # Pad with the designated padding_value (e.g., 0)
        text_ids = np.pad(text_ids, ((0, pad_len)), constant_values=padding_value)
        outs.append(text_ids)

    if not outs:
      return np.array([], dtype=np.int32).reshape(0, max_length)

    stacked_text_ids = np.stack(outs)
    return stacked_text_ids


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks based on estimated character count,
    respecting sentence boundaries where possible.
    Max_chars is an estimate, actual byte length might vary.
    """
    chunks = []
    current_chunk = ""
    # More robust sentence splitting for English and Chinese
    sentences = re.split(r'(?<=[.?!;；。？！])\s*', text)
    # Filter out empty strings that can result from splitting
    sentences = [s for s in sentences if s]

    if not sentences:
        if text: # Handle case where text has no sentence-ending punctuation
            sentences = [text]
        else:
            return [] # No text, no chunks

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Estimate length (simple char count, pinyin will expand this later)
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + " " # Add space between sentences
        else:
            # If adding the sentence exceeds max_chars
            if current_chunk: # Add the previous chunk if it exists
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " " # Start new chunk with current sentence
            else: # Sentence itself is longer than max_chars
                # Simple split for very long sentences (could be improved)
                parts = [sentence[i:i+max_chars] for i in range(0, len(sentence), max_chars)]
                chunks.extend(p.strip() + (" " if i < len(parts)-1 else "") for i, p in enumerate(parts))
                current_chunk = "" # Reset current chunk

    if current_chunk: # Add the last chunk
        chunks.append(current_chunk.strip())

    # Filter out any potential empty chunks again
    chunks = [c for c in chunks if c]
    return chunks


def lens_to_mask(t: np.ndarray, length: int) -> np.ndarray:
    # t: array of lengths, shape (b,)
    # length: maximum sequence length
    # returns: mask of shape (b, length)
    if t.ndim == 0: # Handle single length input
        t = t.reshape(1)
    seq = np.arange(length)
    mask = seq < t[:, None]  # Shape: (b, length)
    return mask


# --- Core Diffusion Loop Logic (Unchanged) ---

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
        #drop_audio_cond=True,
    )

    # Classifier-Free Guidance
    guidance_scale = cfg_strength # Use global or Gradio input
    pred = null_pred + guidance_scale * (pred - null_pred)

    # DDIM-like step (simplified Euler)
    latents = latents + (t_prev - t_curr) * pred # This matches the original Euler step
    latents = jnp.array(latents, dtype=latents_dtype) # Recast JAX tracer

    return latents, state, c_ts, p_ts


def run_inference(
    states, latents, cond, decoder_segment_ids, text_embed_cond, text_embed_uncond, c_ts, p_ts, transformer, config, mesh
):
    transformer_state = states # Assuming states only contain transformer state now

    loop_body_p = functools.partial(
        loop_body,
        transformer=transformer,
        cond=cond,
        decoder_segment_ids=decoder_segment_ids,
        text_embed_cond=text_embed_cond,
        text_embed_uncond=text_embed_uncond,
    )

    # Removed partitioning axis rules context, assume handled by sharding annotations or global context
    # with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    # We need the state back if it were being updated (e.g., BatchNorm), but it's not here.
    latents_final, _, _, _ = jax.lax.fori_loop(0, len(c_ts), loop_body_p, (latents, transformer_state, c_ts, p_ts))

    return latents_final

# --- Gradio Inference Function ---

def generate_audio(
    ref_text: str,
    gen_text: str,
    ref_audio_input: Tuple[int, np.ndarray] | str | None,
    num_inference_steps: int = 50,
    guidance_scale: float = 2.0,
    speed_factor: float = 1.0, # <-- Add speed factor parameter
    use_sway_sampling: bool = False, # <-- Add sway sampling parameter
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[int, np.ndarray]:
    """
    Main function called by Gradio interface.
    """
    global cfg_strength
    cfg_strength = guidance_scale # Update global cfg strength from Gradio input

    t_start_total = time.time()
    max_logging.log(f"Starting audio generation... Steps: {num_inference_steps}, CFG: {guidance_scale}, Speed: {speed_factor}, Sway: {use_sway_sampling}")

    # --- Input Validation and Loading ---
    if not ref_text:
        ref_text = DEFAULT_REF_TEXT
        max_logging.log(f"Using default reference text: '{ref_text}'")
        # raise gr.Error("Reference text cannot be empty.")
    if not gen_text:
        raise gr.Error("Generation text cannot be empty.")
    if ref_audio_input is None:
        raise gr.Error("Reference audio is required.")

    # Load reference audio
    if isinstance(ref_audio_input, str): # File path
        try:
            ref_audio, ref_sr = librosa.load(ref_audio_input, sr=TARGET_SR, mono=True)
            max_logging.log(f"Loaded reference audio from path: {ref_audio_input}")
        except Exception as e:
            raise gr.Error(f"Failed to load reference audio: {e}")
    elif isinstance(ref_audio_input, tuple): # Gradio numpy format (sr, data)
        ref_sr, ref_audio = ref_audio_input
        if ref_sr != TARGET_SR:
            max_logging.log(f"Resampling reference audio from {ref_sr} Hz to {TARGET_SR} Hz.")
            ref_audio = librosa.resample(ref_audio.astype(np.float32)/ 32768.0, orig_sr=ref_sr, target_sr=TARGET_SR)
        if ref_audio.ndim > 1:
             ref_audio = np.mean(ref_audio, axis=1) # Ensure mono
        max_logging.log("Loaded reference audio from Gradio input.")
    else:
        raise gr.Error("Invalid reference audio input format.")

    if ref_audio.size == 0:
         raise gr.Error("Reference audio is empty after loading.")

    # --- Preprocessing ---
    t_start_preprocess = time.time()
    max_logging.log("Preprocessing text and audio...")

    # Ensure reference text ends with space if last char is ASCII
    if ref_text and len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    # Estimate character count per second from reference
    ref_duration_sec = len(ref_audio) / TARGET_SR
    if ref_duration_sec < 0.1:
        raise gr.Error("Reference audio is too short (must be at least 0.1 seconds).")

    # Calculate max characters for chunking based on reference speech rate
    # Add a buffer (e.g., 20%) to handle faster speech or estimation errors
    chars_per_sec_ref = len(ref_text.encode("utf-8")) / ref_duration_sec
    # Estimate max duration for generated chunks based on available sequence length
    max_gen_duration_sec = MAX_DURATION_SECS - ref_duration_sec
    if max_gen_duration_sec <= 0:
        raise gr.Error(f"Reference audio duration ({ref_duration_sec:.1f}s) exceeds max allowed duration ({MAX_DURATION_SECS}s).")

    # Estimate max characters per chunk, ensuring it's positive
    # Use a slightly higher estimate chars_per_sec to be conservative
    estimated_max_chars = max(10, int(chars_per_sec_ref * max_gen_duration_sec * 0.8)) # 80% buffer
    max_logging.log(f"Reference: {ref_duration_sec:.1f}s, {len(ref_text)} chars. Estimated max chars/chunk: {estimated_max_chars}")

    gen_text_batches = chunk_text(gen_text, max_chars=estimated_max_chars)
    num_chunks = len(gen_text_batches)
    max_logging.log(f"Split generation text into {num_chunks} chunks.")

    # === NEW: Batch Size Bucketing Logic ===
    if num_chunks == 0:
         raise gr.Error("Text processing resulted in zero valid chunks. Try different text.")
    if num_chunks > MAX_CHUNKS:
        raise gr.Error(f"Too many text chunks ({num_chunks}). Maximum allowed is {MAX_CHUNKS}. Please shorten the 'Text to Generate'.")


    max_logging.log(f"Split generation text into {len(gen_text_batches)} chunks.")

    # === NEW: Batch Size Bucketing Logic ===
    if num_chunks == 0:
         raise gr.Error("Text processing resulted in zero valid chunks. Try different text.")
    if num_chunks > MAX_CHUNKS:
        raise gr.Error(f"Too many text chunks ({num_chunks}). Maximum allowed is {MAX_CHUNKS}. Please shorten the 'Text to Generate'.")

    # Find the target batch size from buckets
    target_batch_size = MAX_CHUNKS # Initialize just in case
    for bucket in BUCKET_SIZES:
        if num_chunks <= bucket:
            target_batch_size = bucket
            break
    
    padded_items_count = target_batch_size - num_chunks
    total_batch_items = target_batch_size # This is the final batch dimension size

    max_logging.log(f"Processing {num_chunks} chunks. Padding to nearest bucket size: {target_batch_size} (adding {padded_items_count} padding items).")
    # === End of Bucketing Logic ===

    batched_text_list_combined = []
    batched_duration_frames = [] # Duration in mel frames (samples // hop_length)
    hop_length = 256 # Must match get_mel
    ref_audio_len_frames = ref_audio.shape[-1] // hop_length + 1

    # Limit reference audio / text to avoid exceeding max sequence length early
    max_ref_frames = int(global_max_sequence_length * 0.6) # Allow ref max 60% of total length
    if ref_audio_len_frames > max_ref_frames:
        max_logging.log(f"Warning: Truncating reference audio from {ref_audio_len_frames} to {max_ref_frames} frames.")
        ref_audio_len_frames = max_ref_frames
        ref_audio = ref_audio[:ref_audio_len_frames * hop_length]
        # Ideally, truncate ref_text too, but estimating byte length -> char mapping is tricky.
        # Simple approximation: truncate proportionally.
        original_ref_text_len = len(ref_text)
        ref_text = ref_text[:int(original_ref_text_len * (max_ref_frames / (ref_audio.shape[-1] // hop_length + 1)))]
        if ref_text and len(ref_text[-1].encode("utf-8")) == 1: # Ensure space again if truncated
             ref_text += " "
        max_logging.log(f"Truncated reference text length: {len(ref_text)}")


    if ref_audio_len_frames >= global_max_sequence_length:
         raise gr.Error(f"Reference audio ({ref_audio_len_frames} frames) already exceeds max sequence length ({global_max_sequence_length}). Please use shorter audio.")



     # === MODIFIED Duration Estimation Loop ===
    for i, single_gen_text in enumerate(gen_text_batches):
        text_combined = ref_text + single_gen_text
        batched_text_list_combined.append(text_combined)

        # Estimate duration: ref_frames + proportional based on text length estimate
        ref_text_byte_len = len(ref_text.encode('utf-8'))
        gen_text_byte_len = len(single_gen_text.encode('utf-8'))

        # Avoid division by zero if ref_text is empty (shouldn't happen with checks)
        if ref_text_byte_len > 0:
             estimated_gen_frames = int(ref_audio_len_frames / ref_text_byte_len * gen_text_byte_len / speed_factor)
        else:
             # Fallback: estimate based on average chars/sec if ref_text was empty
             avg_chars_per_sec = 5 * speed_factor # A rough guess
             estimated_gen_frames = int(gen_text_byte_len * (TARGET_SR / hop_length) / avg_chars_per_sec) if avg_chars_per_sec > 0 else 50 # Avoid div by zero

        estimated_gen_frames = max(0, estimated_gen_frames)
        # Total duration: ref + estimated gen
        duration_frames = ref_audio_len_frames + estimated_gen_frames
        # Clamp duration to max sequence length
        duration_frames = min(global_max_sequence_length, duration_frames)
        # Ensure duration is at least the length of the reference audio part
        duration_frames = max(ref_audio_len_frames + 1, duration_frames) # Need at least one frame for generation

        batched_duration_frames.append(duration_frames)
        max_logging.log(f"Chunk {i+1}/{len(gen_text_batches)}: Combined text len: {len(text_combined)}, Estimated total frames: {duration_frames}")


    # Convert text to pinyin/chars list
    # This step can be slow, especially for long texts
    pinyin_start_time = time.time()
    final_text_list_pinyin = convert_char_to_pinyin(batched_text_list_combined)
    max_logging.log(f"Pinyin conversion took {time.time() - pinyin_start_time:.2f}s")


    # === Apply Padding to text_ids ===
    # Tokenize the *actual* chunks first
    text_ids_unpadded = list_str_to_idx(final_text_list_pinyin, global_vocab_char_map, max_length=global_max_sequence_length)
    # Now pad to the target_batch_size
    text_ids = np.pad(text_ids_unpadded, ((0, padded_items_count), (0, 0)), constant_values=0)
    # ================================

    # (Condition preparation - uses total_batch_items)
    # ... ref_audio_padded, cond calculation ...
    ref_audio_padded = np.pad(ref_audio, (0, max(0, global_max_sequence_length * hop_length + hop_length - ref_audio.shape[0])))
    ref_audio_padded = ref_audio_padded[np.newaxis, :]
    cond = jitted_get_mel(ref_audio_padded)
    cond_pad_len = global_max_sequence_length - cond.shape[1]
    if cond_pad_len > 0:
        cond = np.pad(cond, ((0,0), (0, cond_pad_len), (0,0)))
    elif cond_pad_len < 0:
        cond = cond[:, :global_max_sequence_length, :]
    # Broadcast condition to the final target batch size
    cond = np.repeat(cond, total_batch_items, axis=0)

    # === Apply Padding to duration_frames_arr ===
    # Use a safe padding value (e.g., min possible duration) for padding items
    safe_padding_duration = ref_audio_len_frames + 1
    padded_durations = batched_duration_frames + [safe_padding_duration] * padded_items_count
    duration_frames_arr = np.array(padded_durations, dtype=np.int32)
    # ==========================================

    # (Mask Calculation - uses total_batch_items and padded arrays)
    # Create ref_len array for the total batch size
    ref_len_frames_arr = np.array([ref_audio_len_frames] * total_batch_items, dtype=np.int32)

    # Ensure padded duration array elements don't exceed max length and are >= ref length + 1
    duration_frames_arr = np.minimum(duration_frames_arr, global_max_sequence_length)
    duration_frames_arr = np.maximum(duration_frames_arr, ref_len_frames_arr + 1)

    # Calculate text lengths based on the *padded* text_ids
    text_lens = np.minimum((text_ids != 0).sum(axis=-1), global_max_sequence_length)


    # Calculate final duration using padded arrays
    effective_min_len = np.maximum(text_lens, ref_len_frames_arr) + 1
    duration_final = np.maximum(effective_min_len, duration_frames_arr)
    duration_final = np.minimum(duration_final, global_max_sequence_length) # Final cap

    # Create masks using final calculated lengths
    cond_mask = lens_to_mask(ref_len_frames_arr, length=global_max_sequence_length) # Mask for reference audio part
    decoder_mask = lens_to_mask(duration_final, length=global_max_sequence_length)  # Mask for the whole sequence generation

    # Prepare segment IDs
    text_decoder_segment_ids = (text_ids != 0).astype(np.int32) # Mask based on text tokens
    decoder_segment_ids = decoder_mask.astype(np.int32)        # Mask based on calculated total duration

    # Apply conditional mask (uses broadcasted cond and cond_mask)
    step_cond = np.where( cond_mask[..., np.newaxis], cond, np.zeros_like(cond) )

    # --- Shard data ---
    step_cond = jax.device_put(step_cond, global_data_sharding)
    text_ids = jax.device_put(text_ids, global_data_sharding)
    decoder_segment_ids = jax.device_put(decoder_segment_ids, global_data_sharding)
    text_decoder_segment_ids = jax.device_put(text_decoder_segment_ids, global_data_sharding)
    cond_mask_sharded = jax.device_put(cond_mask, global_data_sharding) # Shard this too for final masking


    t_end_preprocess = time.time()
    max_logging.log(f"Preprocessing finished in {t_end_preprocess - t_start_preprocess:.2f}s.")
    #get_memory_allocations()

    # --- Text Embedding ---
    t_start_embed = time.time()
    max_logging.log("Generating text embeddings...")
    rng_embed = jax.random.key(global_config.seed + 1) # Use a different seed
    rngs_embed = {'params': rng_embed, 'dropout': rng_embed}

    text_embed_cond = global_jitted_text_encode_funcs[target_batch_size]({"params": global_text_encoder_params},
                                          text_ids,
                                          text_decoder_segment_ids,
                                         rngs_embed)

    # Unconditional embeddings (zero text input)
    text_embed_uncond = global_jitted_text_encode_funcs[target_batch_size]({"params": global_text_encoder_params},
                                  np.zeros_like(text_ids),
                                  text_decoder_segment_ids, # Use zero mask too
                                  rngs_embed)
    t_end_embed = time.time()
    max_logging.log(f"Text embedding generation took {t_end_embed - t_start_embed:.2f}s.")
    #get_memory_allocations()


    # --- Diffusion Sampling ---
    t_start_diffusion = time.time()
    max_logging.log(f"Starting diffusion sampling with {num_inference_steps} steps...")

    # Initial noise (latents)
    latents_shape = (total_batch_items, global_max_sequence_length, 100) # Get latent_dim from model
    latents_rng = jax.random.key(global_config.seed + 2)
    latents = jax.random.normal(latents_rng, latents_shape, dtype=jnp.float32)
    latents = jax.device_put(latents, global_data_sharding)

    # === MODIFIED Timestep Calculation ===
    t_start = 0.0
    timesteps = np.linspace(t_start, 1.0, num_inference_steps + 1).astype(np.float32)

    if use_sway_sampling:
        # Get coefficient from config, default to 0.0 if not found
        sway_coef = global_config.sway_sampling_coef
        if sway_coef is not None:
            max_logging.log(f"Applying Sway Sampling with coefficient: {sway_coef}")
            timesteps = timesteps + sway_coef * (np.cos(np.pi / 2 * timesteps) - 1 + timesteps)
            # Clip timesteps to ensure they remain in [0, 1] after adjustment
            timesteps = np.clip(timesteps, 0.0, 1.0)
        else:
            max_logging.log("Sway sampling enabled but coefficient is 0 or missing in config. Skipping.")
    else:
        max_logging.log("Sway sampling disabled.")

    c_ts = timesteps[:-1] # Current timesteps
    p_ts = timesteps[1:]  # Previous timesteps (sigma_{t-1}, sigma_t in DDIM terms if reversed)
    # === End of Modified Timestep Calculation ===

    # Run inference loop (using pre-compiled partial function)
    y_final_latents = global_p_run_inference_funcs[target_batch_size](
        global_transformer_state, # Pass state
        latents,
        step_cond,
        decoder_segment_ids,
        text_embed_cond,
        text_embed_uncond,
        c_ts,
        p_ts
    )

    # Ensure computation happens
    y_final_latents.block_until_ready()
    t_end_diffusion = time.time()
    max_logging.log(f"Diffusion sampling finished in {t_end_diffusion - t_start_diffusion:.2f}s.")
    #get_memory_allocations()


    # --- Postprocessing (Vocoder) ---
    t_start_post = time.time()
    max_logging.log("Applying Vocoder...")

    # Combine condition and generated parts
    # Use sharded cond_mask here
    out_latents = jnp.where(cond_mask_sharded[..., jnp.newaxis], cond, y_final_latents)

    # Apply Vocoder
    vocoder_rng = jax.random.key(global_config.seed + 3)
    rngs_vocoder = {'params': vocoder_rng, 'dropout': vocoder_rng} # Vocos might need dropout rng
    # Vocoder expects (batch, seq_len, mel_bins)
    # Apply on device
    audio_out_jax = global_jitted_vocos_apply_funcs[target_batch_size]({"params": global_vocos_params}, out_latents, rngs_vocoder)
    audio_out_jax.block_until_ready() # Wait for vocoder to finish

    # Transfer *only the necessary data* to CPU
    max_logging.log("Transferring generated audio to CPU...")
    # Get lengths needed on CPU *before* slicing on device if possible
    # Use the originally calculated durations (before padding)
    cpu_durations = np.array(batched_duration_frames)
    cpu_ref_len_frames = ref_audio_len_frames # Use the (potentially truncated) ref length

    # Transfer all generated audio data for the valid chunks
    audio_out_cpu = np.asarray(audio_out_jax[:num_chunks])

    t_end_post = time.time()
    max_logging.log(f"Vocoder and transfer took {t_end_post - t_start_post:.2f}s.")
    #get_memory_allocations()


    # --- Final Audio Stitching ---
    t_start_stitch = time.time()
    max_logging.log("Stitching audio chunks...")
    final_audio_segments = []

    # Convert frame lengths to sample lengths
    ref_len_samples = cpu_ref_len_frames * hop_length

    for i in range(num_chunks):
        # Get the duration for this specific chunk in frames
        current_duration_frames = cpu_durations[i]
        # Convert duration to samples
        current_duration_samples = current_duration_frames * hop_length

        # Extract the generated part for this chunk
        # Slice from end of reference audio up to the total duration for this chunk
        # audio_out_cpu[i] has shape (seq_len * hop_length,) approx
        generated_part = audio_out_cpu[i, ref_len_samples:current_duration_samples]
        final_audio_segments.append(generated_part)

    # Concatenate all generated segments
    final_audio = np.concatenate(final_audio_segments) if final_audio_segments else np.array([], dtype=np.float32)

    t_end_stitch = time.time()
    max_logging.log(f"Audio stitching took {t_end_stitch - t_start_stitch:.2f}s.")

    t_end_total = time.time()
    total_duration = t_end_total - t_start_total
    generated_audio_duration = len(final_audio) / TARGET_SR
    max_logging.log(f"Total generation time: {total_duration:.2f}s for {generated_audio_duration:.2f}s of audio.")
    if generated_audio_duration > 0:
        rtf = total_duration / generated_audio_duration
        max_logging.log(f"Real-Time Factor (RTF): {rtf:.3f}")


    # Return in Gradio audio format
    return (TARGET_SR, final_audio)


# --- Setup Function ---
def setup_models_and_state(config):
    """
    Initializes models, states, JIT compiles functions, etc.
    Called once when the Gradio app starts.
    """
    global global_config, global_mesh, global_transformer, global_transformer_state
    global global_transformer_state_shardings, global_text_encoder, global_text_encoder_params
    global global_jitted_text_encode_funcs, global_vocos_model, global_vocos_params
    global global_jitted_vocos_apply_funcs, global_vocab_char_map, global_vocab_size
    global global_p_run_inference_funcs, global_data_sharding, global_max_sequence_length
    global jitted_get_mel




    t_start_setup = time.time()
    max_logging.log("Starting one-time setup...")
    global_config = config # Store config globally

    flash_block_sizes = get_flash_block_sizes(config)
    # Store max sequence length from config
    global_max_sequence_length = config.max_sequence_length
    max_logging.log(f"Model configured for max sequence length: {global_max_sequence_length}")

    rng = jax.random.key(config.seed)
    devices_array = create_device_mesh(config)
    global_mesh = Mesh(devices_array, config.mesh_axes)
    mesh = global_mesh # Use local variable for clarity in setup

    if not config.mesh_axes: raise ValueError("config.mesh_axes must be defined (e.g., ['data'])")
    data_axis_name = config.mesh_axes[0]
    model_axis_names = config.mesh_axes[1:]
    max_logging.log(f"Using mesh axes: {config.mesh_axes} (Data axis: '{data_axis_name}')")
    # Determine batch size based on devices
    #num_devices = len(jax.devices())
    #global_batch_size = config.per_device_batch_size * num_devices
    #max_logging.log(f"Using global batch size: {global_batch_size} ({config.per_device_batch_size} per device)")
        # --- Define Basic Sharding Specs ---
    # (Keep existing specs: batch_only, batch_seq, batch_seq_dim, replicated)
    sharding_spec_batch_only = P(data_axis_name)
    sharding_spec_batch_seq = P(data_axis_name, None)
    sharding_spec_batch_seq_dim = P(data_axis_name, None, None)
    #sharding_spec_replicated = P()
    # === NEW: Sharding spec for get_mel input/output ===
    # Input y: (Batch, AudioLength) -> Shard AudioLength along data_axis_name
    sharding_spec_get_mel_input = P(None, data_axis_name)
    # Output spec: (Batch, MelSeqLength, NumMels) -> Shard MelSeqLength along data_axis_name
    sharding_spec_get_mel_output = P(None, data_axis_name, None)
    # =================================================

    # --- JIT Compile get_mel with Sharding ---
    max_logging.log("Compiling get_mel with sharding...")
    # Define the sharding for the input 'y'
    get_mel_in_shardings = (jax.sharding.NamedSharding(mesh, sharding_spec_get_mel_input),) # Tuple for positional args
    # Define the sharding for the output spectrogram
    #get_mel_out_shardings = jax.sharding.NamedSharding(mesh, sharding_spec_replicated)
    get_mel_out_shardings = None

    # Create the jitted function with sharding info
    # Static argnums remain the same as they refer to non-JAX array arguments
    jitted_get_mel = jax.jit(
        get_mel,
        static_argnums=(1, 2, 3, 4, 5, 6, 8), # n_mels, n_fft, etc.
        in_shardings=get_mel_in_shardings,
        out_shardings=get_mel_out_shardings
    )
        # Warmup/Pre-compile get_mel
    try:
        # Determine dummy audio length based on max sequence length and hop
        hop_length = 256 # Should match the default in get_mel
        # Use a length that aligns reasonably with max_sequence_length for the mel output
        # This ensures the sharded dimension is meaningful during compilation.
        # Add some buffer just in case.
        dummy_audio_len = global_max_sequence_length * hop_length + hop_length
        # Use a minimal batch size for compilation, as batch is not sharded here
        compile_batch_size = 1
        dummy_audio_shape = (compile_batch_size, dummy_audio_len)
        dummy_audio = jnp.zeros(dummy_audio_shape, dtype=jnp.float32)
        dummy_audio_sharded = jax.device_put(dummy_audio, get_mel_in_shardings[0])

        max_logging.log(f"Warming up jitted_get_mel with dummy shape {dummy_audio_shape} sharded as {sharding_spec_get_mel_input}...")
        _ = jitted_get_mel(dummy_audio_sharded).block_until_ready()
        max_logging.log("jitted_get_mel successfully compiled and warmed up.")
        # You could inspect the shape and sharding of the output here if needed
        # test_output = jitted_get_mel(dummy_audio_sharded)
        # print("get_mel output shape:", test_output.shape)
        # print("get_mel output sharding:", test_output.sharding)

    except Exception as e:
        max_logging.error(f"Failed to pre-compile/warmup jitted_get_mel with sharding: {e}", exc_info=True)
        # Decide if this is fatal or if the app can continue with non-sharded/later JIT
        raise # Re-raise to make it fatal during setup

    # --- Load Transformer ---
    max_logging.log("Loading F5 Transformer model...")


    global_transformer = F5Transformer2DModel(
        mesh=mesh,
        #latent_dim=config.latent_dim, # Make sure latent_dim is in config
        # text_dim=config.embed_dim, # Make sure embed_dim is in config
        # num_layers=config.num_layers, # Make sure num_layers is in config
        # num_heads=config.num_heads, # Make sure num_heads is in config
        # mlp_ratio=config.mlp_ratio, # Make sure mlp_ratio is in config
        #split_head_dim=config.split_head_dim, # Optional
        attention_kernel=config.attention,
        flash_block_sizes=flash_block_sizes,
        dtype=config.activations_dtype,
        weights_dtype=config.weights_dtype,
        precision=get_precision(config),
        #max_seq_len=global_max_sequence_length # Pass max length here
    )
    transformer = global_transformer # Local var

    # Load weights
    transformer_params, text_encoder_params_loaded = convert_f5_state_dict_to_flax(
        config.pretrained_model_name_or_path, use_ema=config.use_ema
    )
    global_text_encoder_params = flax.core.frozen_dict.FrozenDict(text_encoder_params_loaded) # Store globally

    weights_init_fn = functools.partial(transformer.init_weights, rngs=rng, max_sequence_length=config.max_sequence_length, eval_only=False)
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
    # --- Load Text Encoder ---
    max_logging.log("Loading Text Encoder model...")
    # Infer text_num_embeds from vocab size if possible, or set in config
    global_vocab_char_map, global_vocab_size = get_tokenizer(config.vocab_name_or_path, "custom")

    # Make sure these are in config or have defaults
    # text_dim = config.get("text_dim", 512)
    # conv_layers = config.get("text_conv_layers", 4)

    global_text_encoder = F5TextEmbedding(
        text_num_embeds=2545, # Add 1 if using +1 shift in list_str_to_idx (or adjust tokenizer/model)
        text_dim=512,
        conv_layers=4,
        # Add other necessary params from F5TextEmbedding definition
        dtype=jnp.float32
    )
    text_encoder = global_text_encoder # Local var

    # JIT the text encoder apply function
    # Need dummy inputs
    

    rng_init = jax.random.key(config.seed + 10)
    rngs_init = {'params': rng_init, 'dropout': rng_init}

    # Define sharding for text encoder params (usually replicated)
    #text_encoder_params_sharding = jax.tree_map(lambda x: P(), global_text_encoder_params)
    #text_encoder_params_sharding = jax.tree_map(lambda x: sharding_spec_replicated, global_text_encoder_params)
    global_text_encoder_params = jax.device_put(global_text_encoder_params, None)
    max_logging.log("Text encoder params replicated on devices.")

    text_encode_in_shardings = (
        None, # Params (replicated)
        jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq),      # text (batch sharded)
        jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq),
        None       # text_decoder_segment_ids (batch sharded)
        # RNGs are implicitly handled by JAX, often replicated
    )
    # Define output sharding (usually replicated or matches consumer needs)
    # Assuming output might be replicated or used on host later
    text_encode_out_shardings = jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq_dim)
    def wrap_text_encoder_apply(params,text_ids,text_decoder_segment_ids,rngs):
        return text_encoder.apply(params,text_ids,text_decoder_segment_ids,rngs=rngs)
    global_jitted_text_encode_funcs = {}
    # Compile it once
    for bucket in BUCKET_SIZES:
        global_jitted_text_encode_funcs[bucket] = jax.jit(
            wrap_text_encoder_apply,
            in_shardings=text_encode_in_shardings, # Note the tuple structure for args tree
            out_shardings=text_encode_out_shardings,
            static_argnums=() # No static args in apply needed here
        )
        dummy_text_ids_shape = (bucket, global_max_sequence_length)
        dummy_text_ids = jnp.zeros(dummy_text_ids_shape, dtype=jnp.int32)
        dummy_text_seg_ids = jnp.zeros(dummy_text_ids_shape, dtype=jnp.int32)
        _ = global_jitted_text_encode_funcs[bucket]({"params": global_text_encoder_params},
                                    dummy_text_ids,
                                    dummy_text_seg_ids,
                                    rngs_init)

    max_logging.log("Text Encoder JIT compiled.")


    # --- Load Vocoder ---
    max_logging.log("Loading Vocoder model...")
    # Assumes load_model() returns model definition and params
    global_vocos_model, vocos_params_loaded = load_vocos_model(config.vocoder_model_path) # Add vocoder path to config
    vocos_model = global_vocos_model # Local var
    global_vocos_params = flax.core.frozen_dict.FrozenDict(vocos_params_loaded) # Store globally

    # JIT the vocoder apply function
    # Need dummy input (output of diffusion model)
    

    rng_voc_init = jax.random.key(config.seed + 11)
    rngs_voc_init = {'params': rng_voc_init, 'dropout': rng_voc_init}

    # Shard Vocoder Params (Replicated is usually sufficient)
    #vocos_params_sharding = jax.tree_map(lambda x: sharding_spec_replicated, global_vocos_params)
    global_vocos_params = jax.device_put(global_vocos_params, None)
    max_logging.log("Vocoder params replicated on devices.")

    vocos_apply_in_shardings = (
        None, # Params (replicated)
        jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq_dim), # Input latents (batch sharded)
        None,# RNGs implicitly handled
    )

    # Output is (Batch, AudioLen), so shard batch dim
    vocos_apply_out_shardings = jax.sharding.NamedSharding(mesh, sharding_spec_batch_seq) # Assuming AudioLen is like Seq dim
    def wrap_vocos_apply(params,x,rngs):
        return vocos_model.apply(params,x,rngs=rngs)
    global_jitted_vocos_apply_funcs = {}
    # Compile it once
    for bucket in BUCKET_SIZES:
        global_jitted_vocos_apply_funcs[bucket] = jax.jit(
            wrap_vocos_apply,
            in_shardings=vocos_apply_in_shardings,
            out_shardings=vocos_apply_out_shardings,
            static_argnums=()
        )
        dummy_latents_shape = (bucket, global_max_sequence_length, config.n_mels)
        dummy_latents_vocoder = jnp.zeros(dummy_latents_shape, dtype=jnp.float32)
        _ = global_jitted_vocos_apply_funcs[bucket]({"params": global_vocos_params}, dummy_latents_vocoder, rngs_voc_init)
    max_logging.log("Vocoder JIT compiled.")


    # --- Compile Inference Loop ---
    max_logging.log("Compiling main inference loop...")

    # Define data sharding for inputs passed to p_run_inference during execution
    # Usually data-parallel along batch dimension
    # Match sharding used inside generate_audio
    global_data_sharding = jax.sharding.NamedSharding(mesh, P(config.data_sharding[0])) # Assuming first axis is batch for data


    # Define shardings for inputs to run_inference
    # state sharding already defined: global_transformer_state_shardings
    latents_sharding = global_data_sharding
    cond_sharding = global_data_sharding
    decoder_segment_ids_sharding = global_data_sharding
    text_embed_sharding = global_data_sharding # Or P() if replicated output from text_encoder

    # Timesteps are usually replicated
    ts_sharding = None #jax.sharding.NamedSharding(mesh, P())


    # JIT the run_inference function
    # Use functools.partial to fix static arguments like model def, config, mesh
    partial_run_inference = functools.partial(
        run_inference,
        transformer=transformer, # Pass model def
        config=config,
        mesh=mesh,
        # Other args (latents, cond, etc.) will be provided at call time
    )

    # Define input shardings for the *dynamic* arguments of run_inference
    in_shardings_inf = (
        global_transformer_state_shardings, # states
        latents_sharding,                # latents
        cond_sharding,                   # cond
        decoder_segment_ids_sharding,    # decoder_segment_ids
        text_embed_sharding,             # text_embed_cond
        text_embed_sharding,             # text_embed_uncond
        ts_sharding,                     # c_ts
        ts_sharding                      # p_ts
    )
    # Output sharding (final latents) - should match data sharding probably
    out_shardings_inf = latents_sharding



    # Optional: Compile run_inference once (can take time)
    global_p_run_inference_funcs = {}
    try:
        for bucket in BUCKET_SIZES:
            global_p_run_inference_funcs[bucket] = jax.jit(
                partial_run_inference,
                static_argnums=(), # No static args in the partial itself anymore
                in_shardings=in_shardings_inf,
                out_shardings=out_shardings_inf,
            )
            dummy_latents_shape = (bucket, global_max_sequence_length, config.n_mels)
            dummy_text_embed_shape = (bucket, global_max_sequence_length, 512)
            dummy_text_ids_shape = (bucket, global_max_sequence_length)
            dummy_latents = jnp.zeros(dummy_latents_shape, dtype=jnp.float32)
            dummy_cond = jnp.zeros(dummy_latents_shape, dtype=jnp.float32)
            dummy_decoder_segment_ids = jnp.zeros(dummy_text_ids_shape, dtype=jnp.float32)
            dummy_text_embed = jnp.zeros(dummy_text_embed_shape, dtype=jnp.float32)
            dummy_c_ts = jnp.linspace(0.0, 1.0, config.num_inference_steps + 1)[:-1]
            dummy_p_ts = jnp.linspace(0.0, 1.0, config.num_inference_steps + 1)[1:]
            _ = global_p_run_inference_funcs[bucket](
                global_transformer_state,
                dummy_latents,
                dummy_cond,
                dummy_decoder_segment_ids,
                dummy_text_embed,
                dummy_text_embed,
                dummy_c_ts,
                dummy_p_ts
            )

        max_logging.log("Inference loop JIT compiled.")
    except Exception as e:
        max_logging.error(f"Failed to pre-compile inference loop: {e}")


    t_end_setup = time.time()
    max_logging.log(f"One-time setup completed in {t_end_setup - t_start_setup:.2f}s.")
    get_memory_allocations()

# --- Main Execution Logic ---
def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config

    # Perform one-time setup
    try:
        setup_models_and_state(config)
    except Exception as e:
        max_logging.error(f"Fatal error during setup: {e}", exc_info=True)
        print(f"\n\nERROR DURING SETUP: {e}\nCannot launch Gradio app.")
        return # Exit if setup fails

    # --- Create Gradio Interface ---
    css = """
    .audio-container { display: flex; justify-content: center; align-items: center; }
    .transcription-container { margin-top: 10px; }
    footer {visibility: hidden}
    """
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as iface:
        gr.Markdown("## F5 Text-to-Speech Synthesis")
        gr.Markdown(f"Enter reference text, upload reference audio, and provide the text you want to synthesize. Batch size will be automatically adjusted to fit buckets {BUCKET_SIZES} (max {MAX_CHUNKS} chunks).") # Updated description

        with gr.Row():
            with gr.Column():
                ref_text_input = gr.Textbox(label="Reference Text", info="Text corresponding to the reference audio.", value=DEFAULT_REF_TEXT, lines=3)
                ref_audio_input = gr.Audio(label="Reference Audio", type="numpy")
                gen_text_input = gr.Textbox(label="Text to Generate", info="The text you want the model to speak.", lines=5)
                with gr.Row():
                    steps_slider = gr.Slider(minimum=5, maximum=MAX_INFERENCE_STEPS, value=50, step=1, label="Inference Steps", info="More steps take longer but may improve quality.")
                    cfg_slider = gr.Slider(minimum=1.0, maximum=10.0, value=2.0, step=0.1, label="Guidance Scale (CFG)", info="Higher values follow prompts more strictly but can reduce diversity.")
                with gr.Row():
                    speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed Factor", info="Adjust speech rate (1.0 = reference speed).")
                    # === Add Sway Sampling Switch ===
                    sway_sampling_switch = gr.Checkbox(label="Enable Sway Sampling", value=False, info="Modifies timestep schedule (requires sway_sampling_coef > 0 in config).")
                    # ==============================
                submit_btn = gr.Button("Generate Audio", variant="primary")

            with gr.Column(scale=1): # Make right column narrower
                audio_output = gr.Audio(label="Generated Audio", type="numpy")


        # --- Examples (Updated) ---
        gr.Examples(
            examples=[
                [
                    "And maybe read maybe read that book you brought?",
                    "test.mp3", # Replace path
                    "This is a test of the emergency broadcast system.",
                    50, 2.0, 1.0, False # Sway disabled
                ],
                [
                    "I strongly believe that love is one of the only things we have in this world.",
                    "test.mp3", # Replace path
                    "你好，世界！这是一个测试。",
                    50, 2.5, 1.2, True # Sway enabled (if coef>0 in config)
                ],
                 [
                    DEFAULT_REF_TEXT,
                    "test.mp3", # Replace path
                    "The quick brown fox jumps over the lazy dog.",
                    60, 3.0, 0.8, False # Sway disabled
                ],
                 [
                    DEFAULT_REF_TEXT,
                    "test.mp3", # Replace path
                    "Sway sampling can sometimes alter the generation dynamics.",
                    50, 2.0, 1.0, True # Sway enabled (if coef>0 in config)
                ],
            ],
            # Update inputs list order
            inputs=[ref_text_input, gen_text_input, ref_audio_input, steps_slider, cfg_slider, speed_slider, sway_sampling_switch],
            outputs=[audio_output],
            fn=generate_audio,
            cache_examples=False,
        )

        # Update button click inputs list order
        submit_btn.click(
            fn=generate_audio,
            inputs=[ref_text_input, gen_text_input, ref_audio_input, steps_slider, cfg_slider, speed_slider, sway_sampling_switch],
            outputs=[audio_output],
        )

    # Launch the Gradio app
    max_logging.log("Launching Gradio interface...")
    iface.launch(share=True, server_name="0.0.0.0") # Allow external access if needed


if __name__ == "__main__":
  # Make sure to configure paths and other settings in your pyconfig file (e.g., config.yaml)
  # Example required config fields:
  # seed: 0
  # mesh_axes: ['data'] # Or ['data', 'fsdp'] etc.
  # per_device_batch_size: 1
  # max_sequence_length: 4096 # Adjust based on model/memory
  # latent_dim: 100 # Or actual latent dim of your model
  # embed_dim: 512 # Or actual embed dim
  # num_layers: 12
  # num_heads: 8
  # mlp_ratio: 2
  # n_mels: 100 # Mel bins expected by model/vocoder
  # attention: 'local' # Or 'flash', 'dot_product'
  # activations_dtype: 'bfloat16'
  # weights_dtype: 'bfloat16'
  # pretrained_model_name_or_path: '/path/to/your/f5_weights.safetensors'
  # use_ema: False # Or True
  # vocab_name_or_path: '/path/to/your/vocab.txt'
  # vocoder_model_path: '/path/to/your/vocos_model' # Path for jax-vocos load_model
  # data_sharding: ['data'] # Sharding axis name for batch dim
  # logical_axis_rules: [['batch', 'data']] # Example rule
  # gradio_share: False # Set to True to create public link (use with caution)
  jax.config.update("jax_explain_cache_misses", True)
  jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
  app.run(main)