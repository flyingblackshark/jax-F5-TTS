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

import jax
import math
import jax.numpy as jnp
from flax import nnx
from einops import repeat, rearrange
from ...normalization_flax import AdaLayerNormContinuous, AdaLayerNormZero
from ...attention_flax import FlaxF5Attention
from .... import common_types
from ....common_types import BlockSizes
from ...gradient_checkpoint import GradientCheckpointType

AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV


class F5TransformerBlock(nnx.Module):
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

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: int = 1e-6,
        flash_min_seq_length: int = 4096,
        flash_block_sizes: BlockSizes = None,
        mesh: jax.sharding.Mesh = None,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        attention_kernel: str = "dot_product",
        *,
        rngs: nnx.Rngs
    ):

        self.attn_norm = AdaLayerNormZero(
            dim,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.attn = FlaxF5Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qkv_bias=qkv_bias,
            split_head_dim=False,
            dtype=dtype,
            weights_dtype=weights_dtype,
            attention_kernel=attention_kernel,
            mesh=mesh,
            flash_block_sizes=flash_block_sizes,
            rngs=rngs,
        )

        self.ff_norm = nnx.LayerNorm(
            num_features=dim,
            use_bias=False,
            use_scale=False,
            epsilon=eps,
            dtype=dtype,
            param_dtype=weights_dtype,
            rngs=rngs,
        )
        self.ff = nnx.Sequential(
            
                nnx.Linear(
                    in_features=dim,
                    out_features=int(dim * mlp_ratio),
                    use_bias=True,
                    kernel_init=nnx.with_partitioning(
                        nnx.initializers.lecun_normal(), ("embed", "mlp")
                    ),
                    bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
                    dtype=dtype,
                    param_dtype=weights_dtype,
                    precision=precision,
                    rngs=rngs,
                ),
                nnx.gelu,
                nnx.Linear(
                    in_features=dim,
                    out_features=dim,
                    use_bias=True,
                    kernel_init=nnx.with_partitioning(
                        nnx.initializers.lecun_normal(), ("embed", "mlp")
                    ),
                    bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp",)),
                    dtype=dtype,
                    param_dtype=weights_dtype,
                    precision=precision,
                    rngs=rngs,
                ),
            
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def __call__(
        self,
        x,
        temb,
        image_rotary_emb=None,
        decoder_segment_ids=None,
        deterministic: bool = True,
        rngs: nnx.Rngs = None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(
            x, emb=temb
        )

        # Attention.
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            rope=image_rotary_emb,
            decoder_segment_ids=decoder_segment_ids,
            rngs=rngs,
        )

        x = x + gate_msa * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm)
        x = x + gate_mlp * ff_output

        return x


class ConvPositionEmbedding(nnx.Module):

    def __init__(
        self,
        rngs: nnx.Rngs,
        dim: int,
        kernel_size: int = 31,
        groups: int = 16,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,  
    ):
        self.conv1 = nnx.Conv(
                    in_features=dim,
                    out_features=dim,
                    kernel_size=(kernel_size,),
                    padding="SAME",
                    feature_group_count=groups,
                    dtype=dtype,
                    param_dtype=weights_dtype,
                    precision=precision,
                    rngs=rngs,
                )
        self.conv2 = nnx.Conv(
                    in_features=dim,
                    out_features=dim,
                    kernel_size=(kernel_size,),
                    padding="SAME",
                    feature_group_count=groups,
                    dtype=dtype,
                    param_dtype=weights_dtype,
                    precision=precision,
                    rngs=rngs,
                )
        

    def __call__(self, x, mask=None):
        # 如果提供了 mask，则将 mask 扩展一个维度，并将对应位置置 0
        if mask is not None:
            mask_expanded = jnp.expand_dims(mask, axis=-1)  # (b, n, 1)
            x = jnp.where(mask_expanded, x, 0.0)

        x = self.conv1(x)
        x = jax.nn.mish(x)

        if mask is not None:
            x = jnp.where(mask_expanded, x, 0.0)

        x = self.conv2(x)
        x = jax.nn.mish(x)

        if mask is not None:
            x = jnp.where(mask_expanded, x, 0.0)
        return x


class InputEmbedding(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        mel_dim: int,
        text_dim: int,
        out_dim: int,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
    ):
        self.mel_dim = mel_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.precision = precision

        input_dim = mel_dim + mel_dim + text_dim  # x + cond + text_embed
        self.proj = nnx.Linear(
            in_features=input_dim,
            out_features=out_dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.conv_pos_embed = ConvPositionEmbedding(
            dim=out_dim,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(
        self,
        x,
        cond,
        text_embed,
        decoder_segment_ids=None,
        # drop_audio_cond=False
    ):
        # 如果 drop_audio_cond 为 True，则将 cond 置为全 0
        # if drop_audio_cond:
        #     cond = jnp.zeros_like(cond)

        # 将 x, cond, text_embed 在最后一个维度上拼接
        concat_input = jnp.concatenate([x, cond, text_embed], axis=-1)
        x_proj = self.proj(concat_input)
        if decoder_segment_ids is not None:
            x_proj = x_proj * decoder_segment_ids[..., jnp.newaxis]
        # 将卷积位置编码加到投影结果上
        x_out = x_proj + self.conv_pos_embed(x_proj, mask=decoder_segment_ids)
        if decoder_segment_ids is not None:
            x_out = x_out * decoder_segment_ids[..., jnp.newaxis]
        return x_out


class GRN(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.dim = dim
        # Initialize parameters gamma and beta with shape (1, 1, dim)
        self.gamma = nnx.Param(jnp.zeros((1, 1, dim)))
        self.bias = nnx.Param(jnp.zeros((1, 1, dim)))

    def __call__(self, x):
        # Compute L2 norm over the sequence dimension (axis=1) with keepdims
        Gx = jnp.linalg.norm(x, ord=2, axis=1, keepdims=True)
        # Normalize: divide by mean across the feature dimension (axis=-1)
        Nx = Gx / (jnp.mean(Gx, axis=-1, keepdims=True) + 1e-6)
        return self.gamma.value * (x * Nx) + self.bias.value + x


class ConvNeXtV2Block(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        
    ):
        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.dilation = dilation
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.precision = precision

        # Calculate symmetric padding so that output length matches input length.
        # For a kernel size of 7 and dilation d, padding = d*3.
        padding = (dilation * (7 - 1)) // 2

        # Depthwise convolution: we use feature_group_count=dim to apply a separate kernel per channel.
        self.dwconv = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=(7,),
            strides=(1,),
            padding=((padding, padding),),
            feature_group_count=dim,
            input_dilation=(dilation,),
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Layer normalization (applied over the last dimension)
        self.layer_norm = nnx.LayerNorm(
            num_features=dim,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=weights_dtype,
            rngs=rngs,
        )

        # First pointwise (dense) layer
        self.pwconv1 = nnx.Linear(
            in_features=dim,
            out_features=intermediate_dim,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            rngs=rngs,
        )

        # Apply GRN module on the intermediate features
        self.grn = GRN(dim=intermediate_dim, rngs=rngs)

        # Second pointwise (dense) layer
        self.pwconv2 = nnx.Linear(
            in_features=intermediate_dim,
            out_features=dim,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            rngs=rngs,
        )

    def __call__(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.layer_norm(x)
        x = self.pwconv1(x)
        x = nnx.gelu(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


def get_pos_embed_indices(
    start,
    length,
    #max_pos,
    scale=1.0,
):
    # Create a scale tensor of the same shape as start.
    scale = scale * jnp.ones_like(start, dtype=jnp.float32)
    # Compute positions: add an unsqueezed start to the broadcasted arange scaled appropriately.
    pos = start[:, None] + (jnp.arange(length, dtype=jnp.float32)[None, :] * scale[:, None]).astype(jnp.int32)
    # Ensure positions are less than max_pos; otherwise, use max_pos - 1.
    #pos = jnp.where(pos < max_pos, pos, max_pos - 1)
    return pos.astype(jnp.int32)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, theta_rescale_factor: float = 1.0
):
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


class F5TextEmbedding(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        text_num_embeds: int,
        text_dim: int,
        conv_layers: int = 0,
        conv_mult: int = 2,
        theta: int = 1000,
        precompute_max_pos: int = 4096,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
    ):
        self.text_num_embeds = text_num_embeds
        self.text_dim = text_dim
        self.conv_layers = conv_layers
        self.conv_mult = conv_mult
        self.theta = theta
        self.precompute_max_pos = precompute_max_pos
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.precision = precision

        self.text_embed = nnx.Embed(
            num_embeddings=text_num_embeds + 1,
            features=text_dim,
            dtype=dtype,
            param_dtype=weights_dtype,
            rngs=rngs,
        )  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            #self.freqs_cis = precompute_freqs_cis(text_dim, precompute_max_pos)
            text_blocks = nnx.List([])
            for _ in range(conv_layers):
                text_blocks.append(
                    ConvNeXtV2Block(
                        dim=text_dim,
                        intermediate_dim=text_dim * conv_mult,
                        dtype=dtype,
                        weights_dtype=weights_dtype,
                        precision=precision,
                        rngs=rngs,
                    )
                )
            self.text_blocks = text_blocks
        else:
            self.extra_modeling = False

    def __call__(
        self,
        text,
        #seq_len,
        text_decoder_segment_ids,
    ):  # noqa: F722

        batch, text_len = text.shape[0], text.shape[1]
        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            #batch_start = jnp.zeros((batch,))
            # pos_idx = get_pos_embed_indices(
            #     batch_start,
            #     seq_len
            # )
            #text_pos_embed = self.freqs_cis
            text = text + precompute_freqs_cis(self.text_dim, text_len)

            # convnextv2 blocks
            text = text * text_decoder_segment_ids[..., jnp.newaxis]
            for block in self.text_blocks:
                text = block(text)
                text = text * text_decoder_segment_ids[..., jnp.newaxis]

        return text


def exists(val):
    return val is not None


class SinusPositionEmbedding(nnx.Module):
    def __init__(self, dim: int):
        self.dim = dim

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


class TimestepEmbedding(nnx.Module):
    def __init__(
        self,
        dim: int,
        freq_embed_dim: int = 256,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        *,
        rngs: nnx.Rngs
    ):
        self.dim = dim
        self.freq_embed_dim = freq_embed_dim
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.precision = precision

        # 创建 SinusPositionEmbedding 子模块
        self.time_embed = SinusPositionEmbedding(dim=freq_embed_dim)
        # 定义 MLP，两层全连接，中间用 SiLU 激活函数
        self.time_mlp = nnx.Sequential(
            
                nnx.Linear(
                    in_features=freq_embed_dim,
                    out_features=dim,
                    dtype=dtype,
                    param_dtype=weights_dtype,
                    precision=precision,
                    rngs=rngs,
                ),
                jax.nn.silu,
                nnx.Linear(
                    in_features=dim,
                    out_features=dim,
                    dtype=dtype,
                    param_dtype=weights_dtype,
                    precision=precision,
                    rngs=rngs,
                ),
            
        )

    def __call__(self, timestep):
        """
        timestep: 一个 jnp.ndarray，形状通常为 (batch,)
        返回：形状为 (batch, dim) 的时间嵌入
        """
        time_hidden = self.time_embed(timestep)
        time = self.time_mlp(time_hidden)
        return time


class RotaryEmbedding(nnx.Module):
    def __init__(
        self,
        dim: int,
        use_xpos: bool = False,
        scale_base: float = 512.0,
        interpolation_factor: float = 1.0,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
    ):
        self.dim = dim
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        self.interpolation_factor = interpolation_factor
        self.base = base
        self.base_rescale_factor = base_rescale_factor

        base = base * (base_rescale_factor ** (dim / (dim - 2)))



        assert interpolation_factor >= 1.0

        if not use_xpos:
            self.scale = None
        else:
            self.scale = (jnp.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

    def forward_from_seq_len(self, seq_len: int):
        t = jnp.arange(seq_len)
        return self.__call__(t)

    def __call__(self, t: jax.Array, max_pos: int = 4096):

        if t.ndim == 1:
            t = jnp.expand_dims(t, axis=0)
        inv_freq = 1.0 / (
            self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim)
        )
        freqs = (
            jnp.einsum("b i , j -> b i j", t.astype(jnp.float32), inv_freq)
            / self.interpolation_factor
        )
        freqs_complex = jnp.stack([freqs, freqs], axis=-1)
        freqs_complex = rearrange(freqs_complex, "... d r -> ... (d r)")

        if not exists(self.scale):
            return freqs_complex, 1.0

        power = (t - (max_pos // 2)) / self.scale_base
        scale_val = self.scale ** rearrange(power, "... n -> ... n 1")
        scale_complex = jnp.stack([scale_val, scale_val], axis=-1)
        scale_complex = rearrange(scale_complex, "... d r -> ... (d r)")

        return freqs_complex, scale_complex


class F5Transformer2DModel(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        text_dim: int = 512,
        mel_dim: int = 100,
        dim: int = 1024,
        head_dim: int = 64,
        num_depth: int = 22,
        num_heads: int = 16,
        flash_min_seq_length: int = 4096,
        flash_block_sizes: BlockSizes = None,
        mesh: jax.sharding.Mesh = None,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        theta: int = 1000,
        attention_kernel: str = "dot_product",
        eps: float = 1e-6,
        remat_policy: str = "None",
        names_which_can_be_saved: list = [],
        names_which_can_be_offloaded: list = [],
    ):
        self.text_dim = text_dim
        self.mel_dim = mel_dim
        self.dim = dim
        self.head_dim = head_dim
        self.num_depth = num_depth
        self.num_heads = num_heads
        self.flash_min_seq_length = flash_min_seq_length
        self.flash_block_sizes = flash_block_sizes
        self.mesh = mesh
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.precision = precision
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.theta = theta
        self.attention_kernel = attention_kernel
        self.eps = eps

        self.time_embed = TimestepEmbedding(
            dim=dim,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.input_embed = InputEmbedding(
            mel_dim=mel_dim,
            text_dim=text_dim,
            out_dim=dim,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
            rngs=rngs,
        )
        self.rotary_embed = RotaryEmbedding(head_dim)

        @nnx.split_rngs(splits=num_depth)
        @nnx.vmap(
            in_axes=0,
            out_axes=0,
            transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"},
        )
        def init_block(rngs):
            return F5TransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=head_dim,
                attention_kernel=attention_kernel,
                flash_min_seq_length=flash_min_seq_length,
                flash_block_sizes=flash_block_sizes,
                mesh=mesh,
                dtype=dtype,
                weights_dtype=weights_dtype,
                precision=precision,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                rngs=rngs,
            )

        self.gradient_checkpoint = GradientCheckpointType.from_str(remat_policy)
        self.names_which_can_be_offloaded = names_which_can_be_offloaded
        self.names_which_can_be_saved = names_which_can_be_saved

        self.transformer_blocks = init_block(rngs)

        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=dim,
            elementwise_affine=False,
            eps=eps,
            dtype=dtype,
            weights_dtype=weights_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.proj_out = nnx.Linear(
            in_features=dim,
            out_features=mel_dim,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(), ("mlp", None)
            ),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None,)),
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=precision,
            use_bias=True,
            rngs=rngs,
        )

    def __call__(
        self,
        x,  # noised input audio
        cond,  # masked cond audio
        text_embed,  # text
        timestep,  # time step
        decoder_segment_ids,  # mask
        deterministic: bool = True,
        rngs: nnx.Rngs = None,
    ):
        batch, seq_len = x.shape[0], x.shape[1]

        t = self.time_embed(timestep)
        x = (
            self.input_embed(
                x,
                cond,
                text_embed,
                decoder_segment_ids=decoder_segment_ids,
            )
            * decoder_segment_ids[..., jnp.newaxis]
        )
        image_rotary_emb = self.rotary_embed.forward_from_seq_len(seq_len)

        def scan_fn(carry, block):
            hidden_states_carry, rngs_carry = carry
            hidden_states = block(
                hidden_states_carry,
                t,
                image_rotary_emb,
                decoder_segment_ids,
                deterministic,
                rngs_carry,
            )
            new_carry = (hidden_states, rngs_carry)
            return new_carry, None

        rematted_block_forward = self.gradient_checkpoint.apply(
            scan_fn, self.names_which_can_be_saved, self.names_which_can_be_offloaded
        )
        initial_carry = (x, rngs)
        final_carry, _ = nnx.scan(
            rematted_block_forward,
            length=self.num_depth,
            in_axes=(nnx.Carry, 0),
            out_axes=(nnx.Carry, 0),
        )(initial_carry, self.transformer_blocks)
        x, _ = final_carry
        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output
