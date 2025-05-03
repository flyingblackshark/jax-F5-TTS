# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from functools import partial
from typing import List, Optional, Union, Callable,Any

import jax
import jax.numpy as jnp
import math
from transformers import (CLIPTokenizer, FlaxCLIPTextModel, FlaxT5EncoderModel, AutoTokenizer)
from einops import rearrange
from jax.typing import DTypeLike
from chex import Array

from flax.linen import partitioning as nn_partitioning

from maxdiffusion.utils import logging

from ...models import FlaxAutoencoderKL
from ...schedulers import (FlaxEulerDiscreteScheduler)
from ..pipeline_flax_utils import FlaxDiffusionPipeline
from maxdiffusion.models.f5.transformers.transformer_f5_flax import F5Transformer2DModel,F5TextEmbedding


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Set to True to use python for loop instead of jax.fori_loop for easier debugging
DEBUG = False


class F5Pipeline(FlaxDiffusionPipeline):

  def __init__(
      self,
      f5: F5Transformer2DModel,
      text_encoder: F5TextEmbedding,
      #scheduler: FlaxEulerDiscreteScheduler,
      dtype: jnp.dtype = jnp.float32,
      mesh: Optional[Any] = None,
      config: Optional[Any] = None,
      rng: Optional[Any] = None,
  ):
    super().__init__()
    self.dtype = dtype
    self.register_modules(
        f5=f5,
        text_encoder=text_encoder,
    )
    self.mesh = mesh
    self._config = config
    self.rng = rng

  def _generate(
      self, F5_params, vae_params, latents, latent_image_ids, prompt_embeds, txt_ids, vec, guidance_vec, c_ts, p_ts
  ):

    def loop_body(
        step,
        args,
        transformer,
        latent_image_ids,
        prompt_embeds,
        txt_ids,
        vec,
        guidance_vec,
    ):
      latents, state, c_ts, p_ts = args
      latents_dtype = latents.dtype
      t_curr = c_ts[step]
      t_prev = p_ts[step]
      t_vec = jnp.full((latents.shape[0],), t_curr, dtype=latents.dtype)
      pred = transformer.apply(
          {"params": state["params"]},
          hidden_states=latents,
          img_ids=latent_image_ids,
          encoder_hidden_states=prompt_embeds,
          txt_ids=txt_ids,
          timestep=t_vec,
          guidance=guidance_vec,
          pooled_projections=vec,
      ).sample
      latents = latents + (t_prev - t_curr) * pred
      latents = jnp.array(latents, dtype=latents_dtype)
      return latents, state, c_ts, p_ts

    loop_body_p = partial(
        loop_body,
        transformer=self.F5,
        latent_image_ids=latent_image_ids,
        prompt_embeds=prompt_embeds,
        txt_ids=txt_ids,
        vec=vec,
        guidance_vec=guidance_vec,
    )

    vae_decode_p = partial(self.vae_decode, vae=self.vae, state=vae_params, config=self._config)

    with self.mesh, nn_partitioning.axis_rules(self._config.logical_axis_rules):
      latents, _, _, _ = jax.lax.fori_loop(0, len(c_ts), loop_body_p, (latents, F5_params, c_ts, p_ts))
    image = vae_decode_p(latents)
    return image

  def do_time_shift(self, mu: float, sigma: float, t: Array):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

  def get_lin_function(
      self, x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
  ) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

  def time_shift(self, latents, timesteps):
    # estimate mu based on linear estimation between two points
    lin_function = self.get_lin_function(
        x1=self._config.max_sequence_length, y1=self._config.base_shift, y2=self._config.max_shift
    )
    mu = lin_function(latents.shape[1])
    timesteps = self.do_time_shift(mu, 1.0, timesteps)
    return timesteps

  def __call__(self, timesteps: int, f5_params):
    r"""
    The call function to the pipeline for generation.

    Args:
      txt: jnp.array,
      txt_ids: jnp.array,
      vec: jnp.array,
      num_inference_steps: int,
      height: int,
      width: int,
      guidance_scale: float,
      img: Optional[jnp.ndarray] = None,
      shift: bool = False,
      jit (`bool`, defaults to `False`):

    Examples:

    """

    if isinstance(timesteps, int):
      timesteps = jnp.linspace(1, 0, timesteps + 1)

    global_batch_size = 1 * jax.local_device_count()


    c_ts = timesteps[:-1]
    p_ts = timesteps[1:]

    guidance = jnp.asarray([self._config.guidance_scale] * global_batch_size, dtype=jnp.bfloat16)

    images = self._generate(
        f5_params,
        c_ts,
        p_ts,
    )

    images = images
    return images
