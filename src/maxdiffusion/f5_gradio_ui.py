import gradio as gr # Import Gradio
from typing import Sequence, Tuple
from absl import app
import numpy as np
from maxdiffusion import pyconfig, max_logging
import time
import librosa
import jax
from maxdiffusion.utils.pinyin_utils import chunk_text
import flax
# --- Configuration & Constants ---
cfg_strength = 2.0 # Made this a variable, potentially could be a Gradio slider
TARGET_SR = 24000
MAX_INFERENCE_STEPS = 100 # Default inference steps, could be Gradio input
DEFAULT_REF_TEXT = "and there are so many things about humankind that is bad and evil. I strongly believe that love is one of the only things we have in this world."
# === Add Bucket Constants ===
MAX_CHUNKS = 64
# ==========================

# --- JAX/Model Setup (Global Scope for Gradio) ---
# These will be initialized once when the script starts
global_config = None
global_max_sequence_length = 2048
#global_max_sequence_length = None # Will be set during setup
# --- Core Diffusion Loop Logic (Unchanged) ---

def generate_audio(
    ref_text: str,
    gen_text: str,
    ref_audio_input: Tuple[int, np.ndarray] | str | None,
    num_inference_steps: int = 50,
    guidance_scale: float = 2.0,
    speed_factor: float = 1.0, # <-- Add speed factor parameter
    use_sway_sampling: bool = False, # <-- Add sway sampling parameter
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
    max_gen_duration_sec = global_max_sequence_length * 256 / TARGET_SR - ref_duration_sec
    if max_gen_duration_sec <= 0:
        raise gr.Error(f"Reference audio duration ({ref_duration_sec:.1f}s) exceeds max allowed duration ({global_max_sequence_length * 256 / TARGET_SR}s).")

    # Estimate max characters per chunk, ensuring it's positive
    # Use a slightly higher estimate chars_per_sec to be conservative
    estimated_max_chars = max(10, int(chars_per_sec_ref * max_gen_duration_sec * 0.8 * speed_factor)) # 80% buffer
    max_logging.log(f"Reference: {ref_duration_sec:.1f}s, {len(ref_text)} chars. Estimated max chars/chunk: {estimated_max_chars}")

    gen_text_batches = chunk_text(gen_text, max_chars=estimated_max_chars)
    num_chunks = len(gen_text_batches)
    max_logging.log(f"Split generation text into {num_chunks} chunks.")

    if num_chunks == 0:
         raise gr.Error("Text processing resulted in zero valid chunks. Try different text.")
    if num_chunks > MAX_CHUNKS:
        raise gr.Error(f"Too many text chunks ({num_chunks}). Maximum allowed is {MAX_CHUNKS}. Please shorten the 'Text to Generate'.")

    device_count = jax.device_count()
    multiplier = 1
    while device_count * multiplier < num_chunks:
        multiplier <<= 1
    desired_batch_size = device_count * multiplier
    if desired_batch_size > MAX_CHUNKS:
        raise gr.Error(f"Too many text chunks ({num_chunks}). Cannot pad to device_count * 2^n within max {MAX_CHUNKS} chunks.")

    target_batch_size = desired_batch_size
    padded_items_count = target_batch_size - num_chunks

    max_logging.log(f"Processing {num_chunks} chunks. Padding to device-count*2^n: {target_batch_size} (adding {padded_items_count} padding items).")

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

    # Append padding items so batch size is a multiple of devices
    if padded_items_count > 0:
        for _ in range(padded_items_count):
            batched_text_list_combined.append(ref_text)
            batched_duration_frames.append(min(global_max_sequence_length, ref_audio_len_frames + 1))
        max_logging.log(f"Added {padded_items_count} padded items for device alignment (total batch {len(batched_text_list_combined)}).")

    audio_out_jax = global_f5_pipeline(
        prompt=batched_text_list_combined,
        reference_audio=[ref_audio for _ in range(len(batched_text_list_combined))],
        duration=batched_duration_frames,
        max_sequence_length=global_max_sequence_length,
    )
    

    audio_out_jax.block_until_ready() # Wait for vocoder to finish

    # Transfer *only the necessary data* to CPU
    max_logging.log("Transferring generated audio to CPU...")
    # Get lengths needed on CPU *before* slicing on device if possible
    # Use the originally calculated durations (before padding)
    cpu_durations = np.array(batched_duration_frames)
    cpu_ref_len_frames = ref_audio_len_frames # Use the (potentially truncated) ref length

    # Transfer all generated audio data for the valid chunks
    audio_out_cpu = np.asarray(audio_out_jax[:num_chunks])



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
    global global_config
    global global_f5_pipeline

    # --- Load Transformer ---
    max_logging.log("Loading F5 Pipeline...")

    from maxdiffusion.checkpointing.f5_checkpointer import F5Checkpointer
    global_config = config
    checkpoint_loader = F5Checkpointer(global_config, "F5_CHECKPOINT")
    global_f5_pipeline,_,_ = checkpoint_loader.load_checkpoint()

# --- Main Execution Logic ---
def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    flax.config.update('flax_always_shard_variable', False)
    config = pyconfig.config

    # Perform one-time setup
    try:
        setup_models_and_state(config)
    except Exception as e:
        max_logging.error(f"Fatal error during setup: {e}", exc_info=True)
        print(f"\n\nERROR DURING SETUP: {e}\nCannot launch Gradio app.")
        return # Exit if setup fails

    # --- Create Gradio Interface ---
    with gr.Blocks() as iface:
        gr.Markdown("## F5 Text-to-Speech Synthesis")

        with gr.Row():
            with gr.Column():
                ref_text_input = gr.Textbox(label="Reference Text", info="Text corresponding to the reference audio.", value=DEFAULT_REF_TEXT, lines=3)
                ref_audio_input = gr.Audio(label="Reference Audio", type="numpy",value="https://github.com/flyingblackshark/jax-F5-TTS/raw/refs/heads/main/test.mp3")
                gen_text_input = gr.Textbox(label="Text to Generate", info="The text you want the model to speak.", lines=5)
                with gr.Row():
                    steps_slider = gr.Slider(minimum=5, maximum=MAX_INFERENCE_STEPS, value=20, step=1, label="Inference Steps", info="More steps take longer but may improve quality.")
                    cfg_slider = gr.Slider(minimum=1.0, maximum=10.0, value=2.0, step=0.1, label="Guidance Scale (CFG)", info="Higher values follow prompts more strictly but can reduce diversity.")
                with gr.Row():
                    speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed Factor", info="Adjust speech rate (1.0 = reference speed).")
                    # === Add Sway Sampling Switch ===
                    sway_sampling_switch = gr.Checkbox(label="Enable Sway Sampling", value=True, info="Modifies timestep schedule (requires sway_sampling_coef > 0 in config).")
                    # ==============================
                submit_btn = gr.Button("Generate Audio", variant="primary")

            with gr.Column(scale=1): # Make right column narrower
                audio_output = gr.Audio(label="Generated Audio", type="numpy")

        # Update button click inputs list order
        submit_btn.click(
            fn=generate_audio,
            inputs=[ref_text_input, gen_text_input, ref_audio_input, steps_slider, cfg_slider, speed_slider, sway_sampling_switch],
            outputs=[audio_output],
        )

    # Launch the Gradio app
    max_logging.log("Launching Gradio interface...")
    iface.launch() # Allow external access if needed


if __name__ == "__main__":
  app.run(main)
