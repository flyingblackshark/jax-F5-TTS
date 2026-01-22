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
import time
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
    prompt_clean,
)
import librosa
from maxdiffusion.utils.mel_util import get_mel

from maxdiffusion.maxdiffusion_utils import get_dummy_wan_inputs

import qwix
from maxdiffusion.utils.seq_utils import lens_to_mask
from maxdiffusion.models.vocos.vocos import Vocos
from maxdiffusion.models.vocos.vocos_utils import load_vocos


def cast_with_exclusion(path, x, dtype_to_cast):
  """
  Casts arrays to dtype_to_cast, but keeps params from any 'norm' layer in float32.
  """

  exclusion_keywords = [
      "norm",  # For all LayerNorm/GroupNorm layers
      "condition_embedder",  # The entire time/text conditioning module
      "scale_shift_table",  # Catches both the final and the AdaLN tables
  ]

  path_str = ".".join(str(k.key) if isinstance(k, jax.tree_util.DictKey) else str(k) for k in path)

  if any(keyword in path_str.lower() for keyword in exclusion_keywords):
    print("is_norm_path: ", path)
    # Keep LayerNorm/GroupNorm weights and biases in full precision
    return x.astype(jnp.float32)
  else:
    # Cast everything else to dtype_to_cast
    return x.astype(dtype_to_cast)

def _add_sharding_rule(vs: nnx.VariableState, logical_axis_rules) -> nnx.VariableState:
  vs.sharding_rules = logical_axis_rules
  return vs

def create_sharded_logical_vocos_vocoder(
    devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters, restored_checkpoint=None
):
    def create_model(rngs: nnx.Rngs):
        vocos_model = Vocos(
            rngs=rngs,
        )
        return vocos_model
    vocos_model = nnx.eval_shape(create_model, rngs=rngs)
    graphdef, state, rest_of_state = nnx.split(vocos_model, nnx.Param, ...)

    logical_state_spec = nnx.get_partition_spec(state)
    logical_state_sharding = nn.logical_to_mesh_sharding(logical_state_spec, mesh, config.logical_axis_rules)
    logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
    params = state.to_pure_dict()
    state = dict(nnx.to_flat_state(state))
    
    params = load_vocos(config.vocos_vocoder_pretrained_model_name_or_path, params, "cpu")
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), params)
    for path, val in flax.traverse_util.flatten_dict(params).items():
        # if restored_checkpoint:
        #     path = path[:-1]
        sharding = logical_state_sharding[path].value
        if config.replicate_vocos:
          sharding = NamedSharding(mesh, P())
        state[path].value = device_put_replicated(val, sharding)
    state = nnx.from_flat_state(state)
    vocos_model = nnx.merge(graphdef, state, rest_of_state)


    return vocos_model

def create_sharded_logical_text_encoder(
    devices_array: np.array, mesh: Mesh, rngs: nnx.Rngs, config: HyperParameters, restored_checkpoint=None
):

  def create_model(rngs: nnx.Rngs, f5_config: dict):
    f5_text_encoder = F5TextEmbedding(**f5_config, rngs=rngs)
    return f5_text_encoder

  # 1. Load config.
  # if restored_checkpoint:
  #   f5_config = restored_checkpoint["f5_config"]
  # else:
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
  # if restored_checkpoint:
  #   params = restored_checkpoint["f5_text_encoder_state"]
  # else:
  params = load_f5_text_encoder(
      config.f5_text_encoder_pretrained_model_name_or_path, params, "cpu"
  )
  params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)
  for path, val in flax.traverse_util.flatten_dict(params).items():
    # if restored_checkpoint:
    #   path = path[:-1]

    sharding = logical_state_sharding[path].value
    if config.replicate_text_encoder:
      sharding = NamedSharding(mesh, P())
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
  # if restored_checkpoint:
  #   f5_config = restored_checkpoint["f5_config"]
  # else:
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
    new_params = {}
    params = restored_checkpoint["f5_state"]
    for flax_key, tensor in flax.traverse_util.flatten_dict(params).items():
      if isinstance(flax_key, tuple):
        def _tuple_str_to_int(in_tuple):
          out_list = []
          for item in in_tuple:
            try:
              out_list.append(int(item))
            except ValueError:
              out_list.append(item)
          return tuple(out_list)
        flax_key = _tuple_str_to_int(flax_key)
      new_params[flax_key] = tensor
    params = flax.traverse_util.unflatten_dict(new_params)
  else:
    params = load_f5_transformer(
        config.f5_transformer_pretrained_model_name_or_path, params, "cpu", num_layers=f5_config["num_depth"]
    )
    params = jax.tree_util.tree_map_with_path(
      lambda path, x: cast_with_exclusion(path, x, dtype_to_cast=config.weights_dtype), params
    )
  for path, val in flax.traverse_util.flatten_dict(params).items():
    if restored_checkpoint:
      path = path[:-1]
    sharding = logical_state_sharding[path].value
    if config.replicate_transformer:
      sharding = NamedSharding(mesh, P())
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

  def __init__(
      self,
      text_encoder: F5TextEmbedding,
      transformer: F5Transformer2DModel,
      vocos_vocoder: Vocos,
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
      vocos_vocoder = create_sharded_logical_vocos_vocoder(
          devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint
      )
    return vocos_vocoder

  @classmethod
  def from_checkpoint(cls, 
  config: HyperParameters, 
  restored_checkpoint=None, 
  load_transformer=True,
  load_text_encoder=True,
  load_vocos_vocoder=True,
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
      if load_transformer:
        transformer = cls.load_transformer(
            devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint
        )
      if load_text_encoder:
        text_encoder = cls.load_text_encoder(
            devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint
          )
        global_vocab_char_map, _ = get_tokenizer(config.vocab_name_or_path, "custom")
      if load_vocos_vocoder:
        vocos_vocoder = cls.load_vocos_vocoder(
            devices_array=devices_array, mesh=mesh, rngs=rngs, config=config, restored_checkpoint=restored_checkpoint
        )
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

  def get_ref_mel(self, reference_audio: Union[str,List[str], np.ndarray, tuple] ,max_sequence_length : int):
    # Normalize input to a list of audio items (path | ndarray | (sr, data))
    if isinstance(reference_audio, (str, np.ndarray, tuple)):
        audio_items = [reference_audio]
    elif isinstance(reference_audio, list):
        audio_items = reference_audio
    else:
        raise TypeError(f"Input 'reference_audio' must be a str, np.ndarray, (sr, data) tuple, or a list of them, but got {type(reference_audio)}")

    # Drop None entries, compute batch size based on items count
    audio_items = [u for u in audio_items if u is not None]
    batch_size = len(audio_items)

    ref_max_samples = max_sequence_length * 256
    all_lengths = []
    all_ref_audio = []

    # Iterate and load/normalize each audio item
    for item in audio_items:
        # Case 1: path string
        if isinstance(item, str):
            ref_audio, _ = librosa.load(item, sr=24000)
        # Case 2: (sr, data) tuple
        elif isinstance(item, tuple) and len(item) == 2:
            sr, data = item
            # ensure float32 in [-1,1]
            if data.dtype.kind in {"i", "u"}:
                data = data.astype(np.float32) / 32768.0
            else:
                data = data.astype(np.float32)
            # resample if needed
            if sr != 24000:
                ref_audio = librosa.resample(data, orig_sr=sr, target_sr=24000)
            else:
                ref_audio = data
        # Case 3: raw numpy array
        elif isinstance(item, np.ndarray):
            ref_audio = item
            # convert dtype and scale if integer
            if ref_audio.dtype.kind in {"i", "u"}:
                ref_audio = ref_audio.astype(np.float32) / 32768.0
            else:
                ref_audio = ref_audio.astype(np.float32)
            # assume sr=24000 for arrays; user should provide tuple if otherwise
        else:
            raise TypeError(f"Unsupported reference_audio element type: {type(item)}")

        # mono-ize if multi-channel
        if ref_audio.ndim > 1:
            ref_audio = np.mean(ref_audio, axis=-1)
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
      reference_audio : Union[str,List[str], np.ndarray, tuple] = None,
      duration: Union[int,List[int]] = None,
      max_sequence_length: int = 512,
  ):
    s = time.time()
    cond,ref_audio_len = self.get_ref_mel(reference_audio, max_sequence_length)
    max_logging.log(f"f5_pipeline get_ref_mel time: {time.time() - s}")

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
      prompt = [prompt]
    if duration is not None and isinstance(duration, int):
      duration = [duration]

    batch_size = len(prompt)

    s = time.time()
    text_embed_cond,text_embed_uncond = self.encode_prompt(
        prompt=prompt,
        max_sequence_length=max_sequence_length,
    )
    max_logging.log(f"f5_pipeline encode_prompt time: {time.time() - s}")

    s = time.time()
    latents = self.prepare_latents(
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        dtype=jnp.float32,
        rng=jax.random.key(self.config.seed),
    )
    max_logging.log(f"f5_pipeline prepare_latents time: {time.time() - s}")

    s = time.time()
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


    #data_sharding = NamedSharding(self.mesh, P())
    # Using global_batch_size_to_train_on so not to create more config variables
    # if self.config.global_batch_size_to_train_on // self.config.per_device_batch_size == 0:
    data_sharding = NamedSharding(self.mesh, P(*self.config.data_sharding))
    latents = jax.device_put(latents, data_sharding)
    step_cond = jax.device_put(step_cond, data_sharding)
    text_embed_cond = jax.device_put(text_embed_cond, data_sharding)
    text_embed_uncond = jax.device_put(text_embed_uncond, data_sharding)

    graphdef, state, rest_of_state = nnx.split(self.transformer, nnx.Param, ...)

    # if self._config.time_shift:
    #   timesteps = self.time_shift(latents, timesteps)
    t_start = 0
    timesteps = np.linspace(t_start, 1.0, self.config.num_inference_steps + 1).astype(
        np.float32
    )
    timesteps = timesteps + self.config.sway_sampling_coef * (
        np.cos(np.pi / 2 * timesteps) - 1 + timesteps
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
    max_logging.log(f"f5_pipeline misc prep time: {time.time() - s}")

    s = time.time()
    with self.mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      mel_output = p_run_inference(
          graphdef=graphdef,
          sharded_state=state,
          rest_of_state=rest_of_state
      )
    #mel_output.block_until_ready()
    max_logging.log(f"f5_pipeline p_run_inference time: {time.time() - s}")

    s = time.time()
    mel_output = jnp.where(cond_mask[..., jnp.newaxis], cond, mel_output)
    audio = self.vocos_vocoder(mel_output)
    max_logging.log(f"f5_pipeline vocos_vocoder time: {time.time() - s}")
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