# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union, Optional
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import flax
import flax.linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from ...pyconfig import HyperParameters
from ... import max_logging
from ... import max_utils
from ...max_utils import get_flash_block_sizes, get_precision, device_put_replicated
from maxdiffusion.models.f5.transformers.transformer_f5_flax import (
    F5TextEmbedding,
    F5Transformer2DModel,
)
from maxdiffusion.models.f5.f5_utils import load_f5_transformer,load_f5_text_encoder
from maxdiffusion.utils.pinyin_utils import (
    get_tokenizer,
    chunk_text,
    convert_char_to_pinyin,
    list_str_to_idx,
)
import librosa
from maxdiffusion.utils.mel_util import get_mel
from maxdiffusion.utils.import_utils import is_ftfy_available
from maxdiffusion.maxdiffusion_utils import get_dummy_wan_inputs
import html
import re
import qwix
from maxdiffusion.utils.seq_utils import lens_to_mask
import jax_vocos
def basic_clean(text):
  if is_ftfy_available():
    import ftfy

    text = ftfy.fix_text(text)
  text = html.unescape(html.unescape(text))
  return text.strip()


def whitespace_clean(text):
  text = re.sub(r"\s+", " ", text)
  text = text.strip()
  return text


def prompt_clean(text):
  text = whitespace_clean(basic_clean(text))
  return text


def _add_sharding_rule(vs: nnx.VariableState, logical_axis_rules) -> nnx.VariableState:
  vs.sharding_rules = logical_axis_rules
  return vs

def create_sharded_logical_text_encoder(
    devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters, restored_checkpoint=None
):

  def create_model(rngs: nnx.Rngs, f5_config: dict):
    f5_text_encoder = F5TextEmbedding(**f5_config, rngs=rngs)
    return f5_text_encoder

  # 1. Load config.
  if restored_checkpoint:
    f5_config = restored_checkpoint["f5_config"]
  else:
    f5_config = {}

  #f5_config["mesh"] = mesh
  f5_config["dtype"] = config.activations_dtype
  f5_config["weights_dtype"] = config.weights_dtype
  f5_config["precision"] = get_precision(config)
  f5_config["conv_mult"] = config.text_conv_mult
  f5_config["conv_layers"] = config.text_conv_layers
  f5_config["text_dim"] = config.text_dim
  f5_config["text_num_embeds"] = config.text_num_embeds
  # 2. eval_shape - will not use flops or create weights on device
  # thus not using HBM memory.
  p_model_factory = partial(create_model, f5_config=f5_config)
  f5_text_encoder = nnx.eval_shape(p_model_factory, rngs=rngs)
  graphdef, state, rest_of_state = nnx.split(f5_text_encoder, nnx.Param, ...)

  # 3. retrieve the state shardings, mapping logical names to mesh axis names.
  logical_state_spec = nnx.get_partition_spec(state)
  logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
  logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
  params = state.to_pure_dict()
  state = dict(nnx.to_flat_state(state))

  # 4. Load pretrained weights and move them to device using the state shardings from (3) above.
  # This helps with loading sharded weights directly into the accelerators without fist copying them
  # all to one device and then distributing them, thus using low HBM memory.
  if restored_checkpoint:
    params = restored_checkpoint["f5_text_encoder_state"]
  else:
    params = load_f5_text_encoder(
        config.f5_text_encoder_pretrained_model_name_or_path, params, "cpu"
    )
  params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)
  for path, val in flax.traverse_util.flatten_dict(params).items():
    if restored_checkpoint:
      path = path[:-1]
    sharding = logical_state_sharding[path].value
    state[path].value = device_put_replicated(val, sharding)
  state = nnx.from_flat_state(state)

  f5_text_encoder = nnx.merge(graphdef, state, rest_of_state)
  return f5_text_encoder
    
# For some reason, jitting this function increases the memory significantly, so instead manually move weights to device.
def create_sharded_logical_transformer(
    devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters, restored_checkpoint=None
):

  def create_model(rngs: nnx.Rngs, f5_config: dict):
    f5_transformer = F5Transformer2DModel(**f5_config, rngs=rngs)
    return f5_transformer

  # 1. Load config.
  if restored_checkpoint:
    f5_config = restored_checkpoint["f5_config"]
  else:
    f5_config = {}
  f5_config["mesh"] = mesh
  f5_config["dtype"] = config.activations_dtype
  f5_config["weights_dtype"] = config.weights_dtype
  f5_config["attention_kernel"] = config.attention
  f5_config["precision"] = get_precision(config)
  f5_config["flash_block_sizes"] = get_flash_block_sizes(config)
  f5_config["remat_policy"] = config.remat_policy
  f5_config["names_which_can_be_saved"] = config.names_which_can_be_saved
  f5_config["names_which_can_be_offloaded"] = config.names_which_can_be_offloaded
  f5_config["flash_min_seq_length"] = config.flash_min_seq_length
  f5_config["num_depth"] = config.num_depth
  #f5_config["dropout"] = config.dropout
  # 2. eval_shape - will not use flops or create weights on device
  # thus not using HBM memory.
  p_model_factory = partial(create_model, f5_config=f5_config)
  f5_transformer = nnx.eval_shape(p_model_factory, rngs=rngs)
  graphdef, state, rest_of_state = nnx.split(f5_transformer, nnx.Param, ...)

  # 3. retrieve the state shardings, mapping logical names to mesh axis names.
  logical_state_spec = nnx.get_partition_spec(state)
  logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
  logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
  params = state.to_pure_dict()
  state = dict(nnx.to_flat_state(state))

  # 4. Load pretrained weights and move them to device using the state shardings from (3) above.
  # This helps with loading sharded weights directly into the accelerators without fist copying them
  # all to one device and then distributing them, thus using low HBM memory.
  if restored_checkpoint:
    params = restored_checkpoint["f5_transformer_state"]
  else:
    params = load_f5_transformer(
        config.f5_transformer_pretrained_model_name_or_path, params, "cpu", num_layers=f5_config["num_depth"]
    )
  params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)
  for path, val in flax.traverse_util.flatten_dict(params).items():
    if restored_checkpoint:
      path = path[:-1]
    sharding = logical_state_sharding[path].value
    state[path].value = device_put_replicated(val, sharding)
  state = nnx.from_flat_state(state)

  f5_transformer = nnx.merge(graphdef, state, rest_of_state)
  return f5_transformer


@nnx.jit(static_argnums=(1,), donate_argnums=(0,))
def create_sharded_logical_model(model, logical_axis_rules):
  graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
  p_add_sharding_rule = partial(_add_sharding_rule, logical_axis_rules=logical_axis_rules)
  state = jax.tree.map(p_add_sharding_rule, state, is_leaf=lambda x: isinstance(x, nnx.VariableState))
  pspecs = nnx.get_partition_spec(state)
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  model = nnx.merge(graphdef, sharded_state, rest_of_state)
  return model


class F5Pipeline:
  r"""
  Pipeline for text-to-video generation using Wan.

  tokenizer ([`T5Tokenizer`]):
      Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
      specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
  text_encoder ([`T5EncoderModel`]):
      [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
      the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
  transformer ([`WanModel`]):
      Conditional Transformer to denoise the input latents.
  scheduler ([`FlaxUniPCMultistepScheduler`]):
      A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
  vae ([`AutoencoderKLWan`]):
      Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
  """

  def __init__(
      self,
      text_encoder: F5TextEmbedding,
      transformer: F5Transformer2DModel,
      vocos_vocoder: jax_vocos.Vocos,
      global_vocab_char_map: dict,
      devices_array: np.array,
      mesh: Mesh,
      config: HyperParameters,
  ):
    self.text_encoder = text_encoder
    self.transformer = transformer
    self.vocos_vocoder = vocos_vocoder
    self.global_vocab_char_map = global_vocab_char_map
    self.devices_array = devices_array
    self.mesh = mesh
    self.config = config
    self.p_run_inference = None

  @classmethod
  def get_basic_config(cls, dtype):
    rules = [
        qwix.QtRule(
            module_path=".*",  # Apply to all modules
            weight_qtype=dtype,
            act_qtype=dtype,
        )
    ]
    return rules

  @classmethod
  def get_fp8_config(cls, quantization_calibration_method: str):
    """
    fp8 config rules with per-tensor calibration.
    FLAX API (https://flax-linen.readthedocs.io/en/v0.10.6/guides/quantization/fp8_basics.html#flax-low-level-api):
    The autodiff does not automatically use E5M2 for gradients and E4M3 for activations/weights during training, which is the recommended practice.
    """
    rules = [
        qwix.QtRule(
            module_path=".*",  # Apply to all modules
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e4m3fn,
            bwd_use_original_residuals=True,
            disable_channelwise_axes=True,  # per_tensor calibration
            weight_calibration_method=quantization_calibration_method,
            act_calibration_method=quantization_calibration_method,
            bwd_calibration_method=quantization_calibration_method,
        )
    ]
    return rules

  @classmethod
  def get_qt_provider(cls, config: HyperParameters) -> Optional[qwix.QtProvider]:
    """Get quantization rules based on the config."""
    if not getattr(config, "use_qwix_quantization", False):
      return None

    quantization_calibration_method = getattr(config, "quantization_calibration_method", "absmax")
    match config.quantization:
      case "int8":
        return qwix.QtProvider(cls.get_basic_config(jnp.int8))
      case "fp8":
        return qwix.QtProvider(cls.get_basic_config(jnp.float8_e4m3fn))
      case "fp8_full":
        return qwix.QtProvider(cls.get_fp8_config(quantization_calibration_method))
    return None

  @classmethod
  def quantize_transformer(cls, config: HyperParameters, model: F5Transformer2DModel, pipeline: "WanPipeline", mesh: Mesh):
    """Quantizes the transformer model."""
    q_rules = cls.get_qt_provider(config)
    if not q_rules:
      return model
    max_logging.log("Quantizing transformer with Qwix.")

    batch_size = jnp.ceil(config.per_device_batch_size * jax.local_device_count()).astype(jnp.int32)
    latents, prompt_embeds, timesteps = get_dummy_wan_inputs(config, pipeline, batch_size)
    model_inputs = (latents, timesteps, prompt_embeds)
    with mesh:
      quantized_model = qwix.quantize_model(model, q_rules, *model_inputs)
    max_logging.log("Qwix Quantization complete.")
    return quantized_model

  @classmethod
  def load_text_encoder(
      cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters, restored_checkpoint=None
  ):
    with mesh:
      f5_text_encoder = create_sharded_logical_text_encoder(
          devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint
      )
    return f5_text_encoder
  @classmethod
  def load_transformer(
      cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters, restored_checkpoint=None
  ):
    with mesh:
      f5_transformer = create_sharded_logical_transformer(
          devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint
      )
    return f5_transformer
  @classmethod
  def load_vocos_vocoder(
      cls, devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters, restored_checkpoint=None
  ):
    with mesh:
      vocos_vocoder = jax_vocos.load_model(load_path=config.vocos_vocoder_pretrained_model_name_or_path)
    return vocos_vocoder

  @classmethod
  def from_checkpoint(cls, 
  config: HyperParameters, 
  restored_checkpoint=None, 
    ):
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    rng = jax.random.key(config.seed)
    rngs = nnx.Rngs(rng)
    transformer = None
    global_vocab_char_map = None
    text_encoder = None
    vocos_vocoder = None
    with mesh:
      transformer = cls.load_transformer(
          devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint
      )

      text_encoder = cls.load_text_encoder(
          devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint
      )
      vocos_vocoder = cls.load_vocos_vocoder(
          devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint
      )
    global_vocab_char_map, _ = get_tokenizer(config.vocab_name_or_path, "custom")


    return F5Pipeline(
        text_encoder=text_encoder,
        transformer=transformer,
        vocos_vocoder=vocos_vocoder,
        global_vocab_char_map=global_vocab_char_map,
        devices_array=devices_array,
        mesh=mesh,
        config=config,
    )

  def encode_prompt(
      self,
      prompt: Union[str, List[str]],
      max_sequence_length:int
      #prompt_embeds: jax.Array = None,
  ):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    prompt = [u for u in prompt if u is not None]
    prompt = [prompt_clean(u) for u in prompt]
    pinyin_inputs = convert_char_to_pinyin(prompt)
    text_ids,text_ids_mask = list_str_to_idx(pinyin_inputs, self.global_vocab_char_map, max_length=max_sequence_length)
    text_embed_cond  = self.text_encoder(
      text = text_ids,
      text_decoder_segment_ids=text_ids_mask.astype(np.int32),
    )
    text_embed_uncond = self.text_encoder(
        text=jnp.zeros_like(text_ids),
        text_decoder_segment_ids=text_ids_mask.astype(np.int32),
    )
    text_embed_cond = jnp.pad(text_embed_cond,((0,batch_size-text_embed_cond.shape[0]),(0,0),(0,0)))
    text_embed_uncond = jnp.pad(text_embed_uncond,((0,batch_size-text_embed_uncond.shape[0]),(0,0),(0,0)))

    return text_embed_cond,text_embed_uncond

  #   return latents
  def prepare_latents(
      self,
      batch_size: int,
      max_sequence_length:int,
      dtype: jnp.dtype,
      rng: jnp.ndarray,
  ):

    latents = jax.random.normal(rng, (batch_size, max_sequence_length, 100),dtype=dtype)

    return latents

  def get_ref_mel(self, reference_audio: Union[str,List[str]] ,max_sequence_length : int):
    if isinstance(reference_audio, str):
        audio_paths = [reference_audio]
    elif isinstance(reference_audio, list):
        audio_paths = reference_audio
    else:
        raise TypeError(f"Input 'reference_audio' must be a str or List[str], but got {type(reference_audio)}")
    batch_size = len(reference_audio)
    reference_audio = [u for u in reference_audio if u is not None]

    ref_max_samples = max_sequence_length * 256
    all_lengths = []
    all_ref_audio = []

      # 2. Iterate over each audio path
    for path in audio_paths:
        # Load audio file
        ref_audio, _ = librosa.load(path, sr=24000)

        # Calculate the original length in frames (before padding)
        ref_audio_len = ref_audio.shape[-1] // 256 + 1
        all_lengths.append(ref_audio_len)

        # Pad or truncate the audio waveform to the required length.
        # The original code's padding was `ref_max_samples - 256`, let's stick to that logic.
        target_len = ref_max_samples - 256
        if ref_audio.shape[0] > target_len:
            # Truncate if longer
            ref_audio = ref_audio[:target_len]
        else:
            # Pad if shorter
            ref_audio = np.pad(ref_audio, (0, target_len - ref_audio.shape[0]))
        all_ref_audio.append(ref_audio)
        # 3. Compute the mel spectrogram for the processed audio

    all_ref_audio = jnp.asarray(all_ref_audio)
    all_ref_audio = jnp.pad(all_ref_audio,((0,batch_size-all_ref_audio.shape[0]),(0,0)))
    all_mels = get_mel(all_ref_audio)

    # 4. Stack the list of mel spectrograms into a single JAX array (tensor)
    # This creates a new batch dimension at the beginning.
    # stacked_mels = jnp.concatenate(all_mels, axis=0)
    return all_mels, all_lengths
  def __call__(
      self,
      prompt: Union[str, List[str]] = None,
      reference_audio : Union[str,List[str]] = "/home/fbs/jax-F5-TTS/test.mp3",
      duration: Union[int,List[int]] = None,
      max_sequence_length: int = 512,
  ):
    cond,ref_audio_len = self.get_ref_mel(reference_audio, max_sequence_length)

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
      prompt = [prompt]
    if duration is not None and isinstance(duration, int):
      duration = [duration]

    batch_size = len(prompt)

    text_embed_cond,text_embed_uncond = self.encode_prompt(
        prompt=prompt,
        max_sequence_length=max_sequence_length,
    )

    latents = self.prepare_latents(
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        dtype=jnp.float32,
        rng=jax.random.key(self.config.seed),
    )

    mask = lens_to_mask(duration, length=max_sequence_length)
    cond_mask = lens_to_mask(ref_audio_len, length=max_sequence_length)
    cond_mask = np.pad(
        cond_mask,
        ((0, batch_size - cond_mask.shape[0]), (0, max_sequence_length - cond_mask.shape[-1])),
        constant_values=0,
    )
    mask = np.pad(
        mask,
        ((0, batch_size - mask.shape[0]), (0, max_sequence_length - mask.shape[-1])),
        constant_values=0,
    )
    step_cond = np.where(cond_mask[..., np.newaxis], cond, np.zeros_like(cond))


    data_sharding = NamedSharding(self.mesh, P())
    # Using global_batch_size_to_train_on so not to create more config variables
    if self.config.global_batch_size_to_train_on // self.config.per_device_batch_size == 0:
      data_sharding = NamedSharding(self.mesh, P(*self.config.data_sharding))

    text_embed_cond = jax.device_put(text_embed_cond, data_sharding)
    text_embed_uncond = jax.device_put(text_embed_uncond, data_sharding)

    graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)

    # if self._config.time_shift:
    #   timesteps = self.time_shift(latents, timesteps)
    t_start = 0
    timesteps = jnp.linspace(t_start, 1.0, self.config.num_inference_steps + 1).astype(
        jnp.float32
    )
    timesteps = timesteps + self.config.sway_sampling_coef * (
        jnp.cos(jnp.pi / 2 * timesteps) - 1 + timesteps
    ) 

    c_ts = timesteps[:-1]
    p_ts = timesteps[1:]

    p_run_inference = partial(
        run_inference,
        latents=latents,
        cond=step_cond,
        decoder_segment_ids=mask.astype(np.int32),
        text_embed_cond=text_embed_cond,
        text_embed_uncond=text_embed_uncond,
        c_ts=c_ts,
        p_ts=p_ts,
    )

    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      mel_output = p_run_inference(
          graphdef=graphdef,
          sharded_state=state,
          rest_of_state=rest_of_state
      )
    mel_output = jnp.where(cond_mask[..., jnp.newaxis], cond, mel_output)
    audio = self.vocos_vocoder(mel_output)
    return audio

@jax.jit
def loop_body(
    step,
    args,
    graphdef,
    sharded_state,
    rest_of_state,
    cond,
    decoder_segment_ids,
    text_embed_cond,
    text_embed_uncond,
):
    f5_transformer = nnx.merge(graphdef, sharded_state, rest_of_state)
    latents,  c_ts, p_ts = args
    latents_dtype = latents.dtype
    t_curr = c_ts[step]
    t_prev = p_ts[step]
    t_vec = jnp.full((latents.shape[0],), t_curr, dtype=latents.dtype)
    pred = f5_transformer(
        x=latents,
        cond=cond,
        decoder_segment_ids=decoder_segment_ids,
        text_embed=text_embed_cond,
        timestep=t_vec,
    )
    null_pred = f5_transformer(
        x=latents,
        cond=jnp.zeros_like(cond),
        decoder_segment_ids=decoder_segment_ids,
        text_embed=text_embed_uncond,
        timestep=t_vec,
    )
    cfg_strength = 2
    pred = pred + (pred - null_pred) * cfg_strength
    latents = latents + (t_prev - t_curr) * pred
    latents = jnp.array(latents, dtype=latents_dtype)
    return latents, c_ts, p_ts

def run_inference(
    graphdef,
    sharded_state,
    rest_of_state,
    latents,
    cond,
    decoder_segment_ids,
    text_embed_cond,
    text_embed_uncond,
    c_ts,
    p_ts,
):

    loop_body_p = partial(
        loop_body,
        graphdef=graphdef,
        sharded_state=sharded_state,
        rest_of_state=rest_of_state,
        cond=cond,
        decoder_segment_ids=decoder_segment_ids,
        text_embed_cond=text_embed_cond,
        text_embed_uncond=text_embed_uncond,
    )
    latents, _, _ = jax.lax.fori_loop(
        0, len(c_ts), loop_body_p, (latents, c_ts, p_ts)
    )

    return latents