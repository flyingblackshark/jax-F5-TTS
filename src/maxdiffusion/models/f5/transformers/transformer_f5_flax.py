"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from typing import Optional, Tuple
import jax
import math
import jax.numpy as jnp
import flax
import flax.linen as nn
from einops import repeat, rearrange
from ...normalization_flax import AdaLayerNormContinuous, AdaLayerNormZero
from ...attention_flax import FlaxF5Attention
from .... import common_types
from ....common_types import BlockSizes
from ....utils import BaseOutput

AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV

class F5TransformerBlock(nn.Module):
  r"""
  A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

  Reference: https://arxiv.org/abs/2403.03206

  Parameters:
      dim (`int`): The number of channels in the input and output.
      num_attention_heads (`int`): The number of heads to use for multi-head attention.
      attention_head_dim (`int`): The number of channels in each head.
      context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
          processing of `context` conditions.
  """

  dim: int
  num_attention_heads: int
  attention_head_dim: int
  qk_norm: str = "rms_norm"
  eps: int = 1e-6
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  mlp_ratio: float = 4.0
  qkv_bias: bool = False
  attention_kernel: str = "dot_product"

  def setup(self):

    self.attn_norm = AdaLayerNormZero(self.dim, dtype=self.dtype, weights_dtype=self.weights_dtype, precision=self.precision)

    self.attn = FlaxF5Attention(
        query_dim=self.dim,
        heads=self.num_attention_heads,
        dim_head=self.attention_head_dim,
        qkv_bias=self.qkv_bias,
        split_head_dim=False,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        attention_kernel=self.attention_kernel,
        mesh=self.mesh,
        flash_block_sizes=self.flash_block_sizes,
    )

    self.ff_norm = nn.LayerNorm(
        use_bias=False,
        use_scale=False,
        epsilon=self.eps,
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
    )
    self.ff = nn.Sequential(
        [
            nn.Dense(
                int(self.dim * self.mlp_ratio),
                use_bias=True,
                kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
                bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
                dtype=self.dtype,
                param_dtype=self.weights_dtype,
                precision=self.precision,
            ),
            nn.gelu,
            nn.Dense(
                self.dim,
                use_bias=True,
                kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("embed", "mlp")),
                bias_init=nn.with_logical_partitioning(nn.initializers.zeros, ("mlp",)),
                dtype=self.dtype,
                param_dtype=self.weights_dtype,
                precision=self.precision,
            ),
        ]
    )


    # let chunk size default to None
    self._chunk_size = None
    self._chunk_dim = 0

  def __call__(self, x, temb, image_rotary_emb=None,decoder_segment_ids=None):
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=temb)

    # Attention.
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        rope=image_rotary_emb,
        decoder_segment_ids=decoder_segment_ids,
    )

    x = x + gate_msa * attn_output

    norm = self.ff_norm(x) * (1 + scale_mlp) + shift_mlp
    ff_output = self.ff(norm)
    x = x + gate_mlp * ff_output

    return x



class ConvPositionEmbedding(nn.Module):
    dim: int
    kernel_size: int = 31
    groups: int = 16
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: jax.lax.Precision = None

    @nn.compact
    def __call__(self, x, mask=None):
        # 如果提供了 mask，则将 mask 扩展一个维度，并将对应位置置 0
        if mask is not None:
            mask_expanded = jnp.expand_dims(mask, axis=-1)  # (b, n, 1)
            x = jnp.where(mask_expanded, x, 0.0)
        
        # 这里输入 x 的形状假定为 (batch, n, dim)
        # 使用 SAME padding 保持序列长度不变
        x = nn.Conv(
            features=self.dim,
            kernel_size=(self.kernel_size,),
            padding='SAME',
            feature_group_count=self.groups,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,)(x)
        x = jax.nn.mish(x)
        
        if mask is not None:
            x = jnp.where(mask_expanded, x, 0.0)
        
        x = nn.Conv(
            features=self.dim,
            kernel_size=(self.kernel_size,),
            padding='SAME',
            feature_group_count=self.groups,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,)(x)
        x = jax.nn.mish(x)
        
        if mask is not None:
            x = jnp.where(mask_expanded, x, 0.0)
        return x

class InputEmbedding(nn.Module):
    mel_dim: int
    text_dim: int
    out_dim: int
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: jax.lax.Precision = None
    @nn.compact
    def __call__(self, x, 
                 cond, 
                 text_embed,
                 decoder_segment_ids=None,
                  #drop_audio_cond=False
                  ):
        # 如果 drop_audio_cond 为 True，则将 cond 置为全 0
        # if drop_audio_cond:
        #     cond = jnp.zeros_like(cond)
        
        # 将 x, cond, text_embed 在最后一个维度上拼接
        concat_input = jnp.concatenate([x, cond, text_embed], axis=-1)
        x_proj = nn.Dense(
            features=self.out_dim,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,)(concat_input)
        if decoder_segment_ids is not None:
            x_proj = x_proj * decoder_segment_ids[...,jnp.newaxis]
        # 将卷积位置编码加到投影结果上
        x_out = x_proj + ConvPositionEmbedding(dim=self.out_dim,          
                                               dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,)(x_proj,mask=decoder_segment_ids)
        if decoder_segment_ids is not None:
            x_out = x_out * decoder_segment_ids[...,jnp.newaxis]
        return x_out
    
class GRN(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        # Initialize parameters gamma and beta with shape (1, 1, dim)
        gamma = self.param("gamma", lambda rng, shape: jnp.zeros(shape), (1, 1, self.dim))
        beta = self.param("beta", lambda rng, shape: jnp.zeros(shape), (1, 1, self.dim))
        # Compute L2 norm over the sequence dimension (axis=1) with keepdims
        Gx = jnp.linalg.norm(x, ord=2, axis=1, keepdims=True)
        # Normalize: divide by mean across the feature dimension (axis=-1)
        Nx = Gx / (jnp.mean(Gx, axis=-1, keepdims=True) + 1e-6)
        return gamma * (x * Nx) + beta + x

class ConvNeXtV2Block(nn.Module):
    dim: int
    intermediate_dim: int
    dilation: int = 1
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: jax.lax.Precision = None

    @nn.compact
    def __call__(self, x):
        residual = x
        # Calculate symmetric padding so that output length matches input length.
        # For a kernel size of 7 and dilation d, padding = d*3.
        padding = (self.dilation * (7 - 1)) // 2
        # Depthwise convolution: we use feature_group_count=self.dim to apply a separate kernel per channel.
        x = nn.Conv(
            features=self.dim,
            kernel_size=(7,),
            strides=(1,),
            padding=((padding, padding),),
            feature_group_count=self.dim,
            input_dilation=(self.dilation,),
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,
        )(x)
        # Layer normalization (applied over the last dimension)
        x = nn.LayerNorm(epsilon=1e-6,
                dtype=self.dtype,
                param_dtype=self.weights_dtype,)(x)
        # First pointwise (dense) layer
        x = nn.Dense(features=self.intermediate_dim,
                    dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,)(x)
        x = nn.gelu(x,approximate=False)
        # Apply GRN module on the intermediate features
        x = GRN(dim=self.intermediate_dim)(x)
        # Second pointwise (dense) layer
        x = nn.Dense(features=self.dim,
                    dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,)(x)
        return residual + x
    


def get_pos_embed_indices(start, 
                          #length, 
                          max_pos, 
                          scale=1.0):
    # Create a scale tensor of the same shape as start.
    scale = scale * jnp.ones_like(start, dtype=jnp.float32)
    # Compute positions: add an unsqueezed start to the broadcasted arange scaled appropriately.
    pos = start[:, None] + (jnp.arange(max_pos, dtype=jnp.float32)[None, :] * scale[:, None]).astype(jnp.int32)
    # Ensure positions are less than max_pos; otherwise, use max_pos - 1.
    pos = jnp.where(pos < max_pos, pos, max_pos - 1)
    return pos.astype(jnp.int32)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor: float = 1.0):
    # Rescale theta as in the PyTorch version.
    theta = theta * (theta_rescale_factor ** (dim / (dim - 2)))
    
    # Compute the frequencies for half the dimensions.
    # jnp.arange creates a range; specifying dtype=jnp.float32 ensures floating point division.
    freqs_range = jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)]
    freqs = 1.0 / (theta ** (freqs_range / dim))
    
    # Create an array for t.
    t = jnp.arange(end)
    
    # Compute the outer product between t and the frequencies.
    freqs = jnp.outer(t, freqs)
    
    # Compute cosine and sine parts.
    freqs_cos = jnp.cos(freqs)  # real part
    freqs_sin = jnp.sin(freqs)  # imaginary part
    
    # Concatenate the cosine and sine parts along the last dimension.
    return jnp.concatenate([freqs_cos, freqs_sin], axis=-1)

class F5TextEmbedding(nn.Module):

    text_num_embeds:int
    text_dim:int
    conv_layers:int=0
    conv_mult:int=2
    theta:int = 1000
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: jax.lax.Precision = None
    def setup(self):
        self.text_embed = nn.Embed(self.text_num_embeds + 1, self.text_dim, dtype=self.dtype)  # use 0 as filler token
        
        if self.conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.freqs_cis = precompute_freqs_cis(self.text_dim, self.precompute_max_pos)
            self.text_blocks = [ConvNeXtV2Block(
            self.text_dim, self.text_dim * self.conv_mult,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,) for _ in range(self.conv_layers)]
            
        else:
            self.extra_modeling = False

    def __call__(self, 
                 text, 
                 #seq_len,
                 text_decoder_segment_ids):#, drop_text=False):  # noqa: F722
        
        batch, text_len = text.shape[0], text.shape[1]

        # if drop_text:  # cfg for text
        #     text = jnp.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = jnp.zeros((batch,))
            pos_idx = get_pos_embed_indices(batch_start, 
                                            #seq_len, 
                                            max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = text * text_decoder_segment_ids[...,jnp.newaxis]
            for block in self.text_blocks:
                text = block(text)
                text = text * text_decoder_segment_ids[...,jnp.newaxis]

        return text
    def init_weights(self, rngs, max_sequence_length, eval_only=True):
        num_devices = len(jax.devices())
        batch_size = 1 * num_devices
        decoder_segment_ids_shape = (
            batch_size,
            max_sequence_length
        )
        # bs, encoder_input, seq_length
        txt_ids_shape = (
            batch_size,
            max_sequence_length
        )

        text_decoder_segment_ids_shape = (
            batch_size,
            max_sequence_length,
        )
        text_ids = jnp.zeros(txt_ids_shape, dtype=jnp.int32)
        decoder_segment_ids = jnp.zeros(decoder_segment_ids_shape, dtype=jnp.int32)
        text_decoder_segment_ids = jnp.zeros(text_decoder_segment_ids_shape,dtype=jnp.int32)
        if eval_only:
            return jax.eval_shape(
                self.init,
                    rngs,
                    text=text_ids,
                    seq_len=max_sequence_length,
                    decoder_segment_ids=decoder_segment_ids,
                    text_decoder_segment_ids=text_decoder_segment_ids,
            )["params"]
        else:
            return self.init(
                rngs,
                text=text_ids,
                seq_len=max_sequence_length,
                decoder_segment_ids=decoder_segment_ids,
                text_decoder_segment_ids=text_decoder_segment_ids,
            )["params"]
def exists(val):
    return val is not None

class SinusPositionEmbedding(nn.Module):
    dim: int

    def __call__(self, x, scale: float = 1000.0):
        """
        x: 一个 jnp.ndarray，通常形状为 (batch,) 或 (batch, ...)。
        返回：一个形状为 (batch, dim) 的张量，其中 dim = 2 * (self.dim // 2)
        """
        half_dim = self.dim // 2
        # 计算指数衰减的因子
        emb_factor = math.log(10000) / (half_dim - 1)
        # 生成 [0, half_dim) 的数组，并计算对应的指数权重
        emb = jnp.exp(-emb_factor * jnp.arange(half_dim, dtype=x.dtype))
        # 扩展维度后进行乘法
        # 假设 x 的形状为 (batch,) 则 expand_dims(x, axis=-1) 得到 (batch, 1)
        # emb 扩展为 (1, half_dim)
        emb = scale * jnp.expand_dims(x, axis=-1) * jnp.expand_dims(emb, axis=0)
        # 分别计算 sin 与 cos，再在最后一个维度上拼接
        sin_emb = jnp.sin(emb)
        cos_emb = jnp.cos(emb)
        return jnp.concatenate([sin_emb, cos_emb], axis=-1)

class TimestepEmbedding(nn.Module):
    dim: int
    freq_embed_dim: int = 256
    dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    precision: jax.lax.Precision = None

    def setup(self):
        # 创建 SinusPositionEmbedding 子模块
        self.time_embed = SinusPositionEmbedding(dim=self.freq_embed_dim)
        # 定义 MLP，两层全连接，中间用 SiLU 激活函数
        self.linear1 = nn.Dense(
            features=self.dim,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,)
        self.linear2 = nn.Dense(
            features=self.dim,
            dtype=self.dtype,
            param_dtype=self.weights_dtype,
            precision=self.precision,)

    def __call__(self, timestep):
        """
        timestep: 一个 jnp.ndarray，形状通常为 (batch,)
        返回：形状为 (batch, dim) 的时间嵌入
        """
        # 先通过正弦位置嵌入层
        time_hidden = self.time_embed(timestep)
        # 保持数据类型一致（通常不需要额外转换，因为 JAX 会自动处理类型）
        #time_hidden = time_hidden.astype(timestep.dtype)
        # 通过 MLP 获得最终的时间嵌入
        time = self.linear2(nn.silu(self.linear1(time_hidden)))
        return time
    
class RotaryEmbedding(nn.Module):
    dim: int
    use_xpos: bool = False
    scale_base: float = 512.0
    interpolation_factor: float = 1.0
    base: float = 10000.0
    base_rescale_factor: float = 1.0

    def setup(self):
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base = self.base * (self.base_rescale_factor ** (self.dim / (self.dim - 2)))

        self.inv_freq = 1. / (base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))

        assert self.interpolation_factor >= 1.

        if not self.use_xpos:
            self.scale = None # No need for register_buffer('scale', None)
        else:
            self.scale = (jnp.arange(0, self.dim, 2) + 0.4 * self.dim) / (1.4 * self.dim)


    def forward_from_seq_len(self, seq_len: int):
        t = jnp.arange(seq_len)
        return self.__call__(t)

    def __call__(self, t: jax.Array , max_pos:int = 4096):

        if t.ndim == 1:
            t = jnp.expand_dims(t, axis=0)

        freqs = jnp.einsum('b i , j -> b i j', t.astype(jnp.float32), self.inv_freq) / self.interpolation_factor
        freqs_complex = jnp.stack([freqs, freqs], axis=-1)
        freqs_complex = rearrange(freqs_complex, '... d r -> ... (d r)')

        if not exists(self.scale):
            return freqs_complex, 1.

        power = (t - (max_pos // 2)) / self.scale_base
        scale_val = self.scale ** rearrange(power, '... n -> ... n 1')
        scale_complex = jnp.stack([scale_val, scale_val], axis=-1)
        scale_complex = rearrange(scale_complex, '... d r -> ... (d r)')

        return freqs_complex, scale_complex
    
class F5Transformer2DModel(nn.Module):
  flash_min_seq_length: int = 4096
  flash_block_sizes: BlockSizes = None
  mesh: jax.sharding.Mesh = None
  dtype: jnp.dtype = jnp.float32
  weights_dtype: jnp.dtype = jnp.float32
  precision: jax.lax.Precision = None
  mlp_ratio: float = 2.0
  qkv_bias: bool = True
  theta: int = 1000
  attention_kernel: str = "dot_product"
  eps = 1e-6


  drop_text:bool = False
  text_num_embeds:int = 2545
  text_dim:int = 512
  mel_dim:int = 100
  conv_layers:int=4
  dim:int = 1024
  dim_head:int = 64
  depth:int = 22
  heads:int = 16

  def setup(self):
    self.time_embed = TimestepEmbedding(
            dim=self.dim,
            dtype=self.dtype,
            weights_dtype=self.weights_dtype,
            precision=self.precision,)
    self.input_embed = InputEmbedding(
        mel_dim=self.mel_dim,
        text_dim=self.text_dim,
        out_dim=self.dim,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,)
    #self.text_embed = TextEmbedding(self.text_num_embeds, self.text_dim, conv_layers=self.conv_layers)
    self.rotary_embed = RotaryEmbedding(self.dim_head)

    blocks = []
    for _ in range(self.depth):
      block = F5TransformerBlock(
          dim=self.dim,
          num_attention_heads=self.heads,
          attention_head_dim=self.dim_head,
          attention_kernel=self.attention_kernel,
          flash_min_seq_length=self.flash_min_seq_length,
          flash_block_sizes=self.flash_block_sizes,
          mesh=self.mesh,
          dtype=self.dtype,
          weights_dtype=self.weights_dtype,
          precision=self.precision,
          mlp_ratio=self.mlp_ratio,
          qkv_bias=self.qkv_bias,
      )
      blocks.append(block)
    self.blocks = blocks

    self.norm_out = AdaLayerNormContinuous(
        self.dim,
        elementwise_affine=False,
        eps=self.eps,
        dtype=self.dtype,
        weights_dtype=self.weights_dtype,
        precision=self.precision,
    )

    self.proj_out = nn.Dense(
        self.mel_dim,
        kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("mlp", None)),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
        dtype=self.dtype,
        param_dtype=self.weights_dtype,
        precision=self.precision,
        use_bias=True,
    )

  def __call__(
      self,
      x, #noised input audio
      cond, #masked cond audio
      text_embed, #text
      timestep, #time step
      decoder_segment_ids, #mask
      #text_decoder_segment_ids,#text mask
      #drop_text:bool = False,
      #drop_audio_cond:bool = False,
      train: bool = False,
  ):
    batch, seq_len = x.shape[0], x.shape[1]
    
    t = self.time_embed(timestep)
    #if drop_text:  # cfg for text
    #    txt_ids = jnp.zeros_like(txt_ids)
    #text_embed = self.text_embed(txt_ids, seq_len,decoder_segment_ids=decoder_segment_ids,text_decoder_segment_ids=text_decoder_segment_ids, drop_text=self.drop_text)
    #text_embed = nn.with_logical_constraint(text_embed, ("activation_batch", None))
    x = self.input_embed(x,
                         cond,
                         text_embed,
                         decoder_segment_ids=decoder_segment_ids,
                         #drop_audio_cond=drop_audio_cond
                         ) * decoder_segment_ids[...,jnp.newaxis]
    image_rotary_emb = self.rotary_embed.forward_from_seq_len(seq_len)
    #image_rotary_emb = nn.with_logical_constraint(image_rotary_emb, ("activation_batch", "activation_embed"))

    for block in self.blocks:
      x = block(
          x=x,
          temb=t,
          image_rotary_emb=image_rotary_emb,
          decoder_segment_ids=decoder_segment_ids,
      )

    x = self.norm_out(x, t)
    output = self.proj_out(x)
    return output

  def init_weights(self, rngs, max_sequence_length, eval_only=True):
    num_devices = len(jax.devices())
    batch_size = 1 * num_devices
    batch_image_shape = (
        batch_size,
        max_sequence_length,
        100
    )
    decoder_segment_ids_shape = (
        batch_size,
        max_sequence_length
    )
    # bs, encoder_input, seq_length
    text_embed_shape = (
        batch_size,
        max_sequence_length,
        512
    )

    img = jnp.zeros(batch_image_shape, dtype=self.dtype)
    text_embed = jnp.zeros(text_embed_shape, dtype=jnp.int32)
    decoder_segment_ids = jnp.zeros(decoder_segment_ids_shape, dtype=jnp.int32)
    t = jnp.asarray((0,))
    if eval_only:
      return jax.eval_shape(
          self.init,
            rngs,
            x=img,
            cond=img,
            text_embed=text_embed,
            timestep=t,
            decoder_segment_ids=decoder_segment_ids,
      )["params"]
    else:
        return self.init(
            rngs,
            x=img,
            cond=img,
            text_embed=text_embed,
            timestep=t,
            decoder_segment_ids=decoder_segment_ids,
        )["params"]
