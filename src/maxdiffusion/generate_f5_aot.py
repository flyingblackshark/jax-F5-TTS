import pickle
import gradio as gr # Import Gradio
from typing import Callable, List, Union, Sequence, Tuple
from absl import app
from contextlib import ExitStack
import functools
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
from jax_vocos import load_model as load_vocos_model # Renamed to avoid conflict
from jax.experimental.serialize_executable import serialize
from maxdiffusion.utils.mel_util import get_mel
from maxdiffusion.utils.pinyin_utils import get_tokenizer,chunk_text,convert_char_to_pinyin,list_str_to_idx
# --- Configuration & Constants ---
#jax.experimental.compilation_cache.compilation_cache.set_cache_dir("./jax_cache")
cfg_strength = 2.0 # Made this a variable, potentially could be a Gradio slider
TARGET_SR = 24000
MAX_DURATION_SECS = 40 # Maximum duration allowed for reference + generation combined (adjust as needed)

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
def save_compiled(compiled, save_name):
  """Serialize and save the compiled function."""
  serialized, _, _ = serialize(compiled)
  with open(save_name, "wb") as f:
    pickle.dump(serialized, f)
    

def lens_to_mask(t: jnp.ndarray, length: int) -> jnp.ndarray:
    # t: array of lengths, shape (b,)
    # length: maximum sequence length
    # returns: mask of shape (b, length)
    if t.ndim == 0: # Handle single length input
        t = t.reshape(1)
    seq = jnp.arange(length)
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
        compiled_get_mel = jitted_get_mel.lower(dummy_audio_sharded).compile()
        save_compiled(compiled_get_mel, "get_mel_aot.pickle")
        max_logging.log("jitted_get_mel successfully AOT compiled.")
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
        text_encode_compiled = global_jitted_text_encode_funcs[bucket].lower(
        {"params": global_text_encoder_params},
                                    dummy_text_ids,
                                    dummy_text_seg_ids,
                                    rngs_init).compile()
        save_compiled(text_encode_compiled, f"text_encode_aot_{bucket}.pickle")
    max_logging.log("Text Encoder AOT compiled.")


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
    def wrap_text_encoder_apply(params,x,rngs):
        return vocos_model.apply(params,x,rngs=rngs)
    global_jitted_vocos_apply_funcs = {}
    # Compile it once
    for bucket in BUCKET_SIZES:
        global_jitted_vocos_apply_funcs[bucket] = jax.jit(
            wrap_text_encoder_apply,
            in_shardings=vocos_apply_in_shardings,
            out_shardings=vocos_apply_out_shardings,
            static_argnums=()
        )
        dummy_latents_shape = (bucket, global_max_sequence_length, config.n_mels)
        dummy_latents_vocoder = jnp.zeros(dummy_latents_shape, dtype=jnp.float32)
        vocos_apply_compiled = global_jitted_vocos_apply_funcs[bucket].lower({"params": global_vocos_params}, dummy_latents_vocoder, rngs_voc_init).compile()
        save_compiled(vocos_apply_compiled, f"vocos_apply_aot_{bucket}.pickle")
        max_logging.log(f"Batch Size {bucket} Vocos Cost analysis: {vocos_apply_compiled.cost_analysis()}")
        max_logging.log(f"Batch Size {bucket} Vocos Memory analysis: {vocos_apply_compiled.memory_analysis()}")
    max_logging.log("Vocoder AOT compiled.")


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
            dummy_decoder_segment_ids = jnp.zeros(dummy_text_ids_shape, dtype=jnp.int32)
            dummy_text_embed = jnp.zeros(dummy_text_embed_shape, dtype=jnp.float32)
            dummy_c_ts = jnp.linspace(0.0, 1.0, config.num_inference_steps + 1)[:-1]
            dummy_p_ts = jnp.linspace(0.0, 1.0, config.num_inference_steps + 1)[1:]
            run_inference_compiled = global_p_run_inference_funcs[bucket].lower(
                global_transformer_state,
                dummy_latents,
                dummy_cond,
                dummy_decoder_segment_ids,
                dummy_text_embed,
                dummy_text_embed,
                dummy_c_ts,
                dummy_p_ts
            ).compile()
            save_compiled(run_inference_compiled, f"run_inference_aot_{bucket}.pickle")
            max_logging.log(f"Batch Size {bucket} Inference Cost analysis: {run_inference_compiled.cost_analysis()}")
            max_logging.log(f"Batch Size {bucket} Inference Memory analysis: {run_inference_compiled.memory_analysis()}")

        max_logging.log("Inference loop AOT compiled.")
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
  #jax.config.update("jax_explain_cache_misses", True)
  #jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
  app.run(main)