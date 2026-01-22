"""
Copyright 2024 Google LLC

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
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec
import jax.lax
import flax.linen as nn

class AdaLayerNormContinuous(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
    ):
        self.embedding_dim = embedding_dim
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.bias = bias
        self.norm_type = norm_type
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.precision = precision

        self.linear = nnx.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim * 2,
            use_bias=bias,
            dtype=dtype,
            param_dtype=weights_dtype,
            precision=self.precision,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp")),
            rngs=rngs,
        )

        self.layer_norm = nnx.LayerNorm(
            num_features=embedding_dim,
            epsilon=eps,
            use_bias=elementwise_affine,
            use_scale=elementwise_affine,
            dtype=dtype,
            param_dtype=weights_dtype,
            rngs=rngs,
        )

    def __call__(self, x, conditioning_embedding):
        assert self.norm_type == "layer_norm"
        emb = jax.nn.silu(conditioning_embedding)
        emb = self.linear(emb)
        scale, shift = jnp.split(emb, 2, axis=-1)
        shift = nn.with_logical_constraint(shift, ("activation_batch", "activation_embed"))
        scale = nn.with_logical_constraint(scale, ("activation_batch", "activation_embed"))
        x = self.layer_norm(x)
        x = (1 + scale[:, None, :]) * x + shift[:, None, :]
        return x


class AdaLayerNormZero(nnx.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_type: str = "layer_norm",
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        *,
        rngs: nnx.Rngs = None,
    ):
        self.embedding_dim = embedding_dim
        self.norm_type = norm_type
        self.bias = bias
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.precision = precision

        self.linear = nnx.Linear(
            in_features=embedding_dim,
            out_features=6 * embedding_dim,
            use_bias=bias,
            dtype=dtype,
            param_dtype=weights_dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("embed", "mlp")),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, ("mlp")),
            precision=self.precision,
            rngs=rngs,
        )

        self.layer_norm = nnx.LayerNorm(
            num_features=embedding_dim,
            epsilon=1e-6,
            use_bias=False,
            use_scale=False,
            dtype=dtype,
            param_dtype=weights_dtype,
            rngs=rngs,
        )

    def __call__(self, x, emb):
        emb = jax.nn.silu(emb)
        emb = self.linear(emb)
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = jnp.split(
            emb[:, None, :], 6, axis=-1
        )
        shift_msa = nn.with_logical_constraint(shift_msa, ("activation_batch", "activation_embed"))
        scale_msa = nn.with_logical_constraint(scale_msa, ("activation_batch", "activation_embed"))
        gate_msa = nn.with_logical_constraint(gate_msa, ("activation_batch", "activation_embed"))
        shift_mlp = nn.with_logical_constraint(shift_mlp, ("activation_batch", "activation_embed"))
        scale_mlp = nn.with_logical_constraint(scale_mlp, ("activation_batch", "activation_embed"))
        gate_mlp = nn.with_logical_constraint(gate_mlp, ("activation_batch", "activation_embed"))
        if self.norm_type == "layer_norm":
            x = self.layer_norm(x)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({self.norm_type}) provided. Supported ones are: 'layer_norm'."
            )
        x = x * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nnx.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_type: str = "layer_norm",
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision = None,
        *,
        rngs: nnx.Rngs = None,
    ):
        self.embedding_dim = embedding_dim
        self.norm_type = norm_type
        self.bias = bias
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.precision = precision

        self.dense = nnx.Linear(
            features=3 * embedding_dim,
            use_bias=bias,
            dtype=dtype,
            param_dtype=weights_dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            bias_init=jnp.zeros,
        )

        self.layer_norm = nnx.LayerNorm(
            features=embedding_dim,
            epsilon=1e-6,
            use_bias=False,
            use_scale=False,
            dtype=dtype,
            param_dtype=weights_dtype,
        )

    def __call__(self, x, emb):
        emb = jax.nn.silu(emb)
        emb = self.dense(emb, precision=self.precision)
        shift_msa, scale_msa, gate_msa = jnp.split(emb[:, None, :], 3, axis=-1)
        shift_msa = jax.lax.with_sharding_constraint(
            shift_msa, PartitionSpec("activation_batch", "activation_embed")
        )
        scale_msa = jax.lax.with_sharding_constraint(
            scale_msa, PartitionSpec("activation_batch", "activation_embed")
        )
        gate_msa = jax.lax.with_sharding_constraint(
            gate_msa, PartitionSpec("activation_batch", "activation_embed")
        )
        if self.norm_type == "layer_norm":
            x = self.layer_norm(x)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({self.norm_type}) provided. Supported ones are: 'layer_norm'."
            )
        x = x * (1 + scale_msa) + shift_msa
        return x, gate_msa
