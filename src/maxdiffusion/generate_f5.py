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

from safetensors import safe_open
from typing import Callable, List, Union, Sequence
from absl import app
from contextlib import ExitStack
import functools
import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec as P
import jax.numpy as jnp
from flax.linen import partitioning as nn_partitioning
from maxdiffusion import pyconfig, max_logging
from maxdiffusion.models.f5.transformers.transformer_f5_flax import (
    F5TextEmbedding,
    F5Transformer2DModel,
)

from maxdiffusion.max_utils import (
    device_put_replicated,
    get_memory_allocations,
    create_device_mesh,
    get_flash_block_sizes,
    get_precision,
    setup_initial_state,
)
import time
from maxdiffusion.models.f5.f5_utils import load_f5_transformer,load_f5_text_encoder
from maxdiffusion.utils.mel_util import get_mel
from maxdiffusion.utils.pinyin_utils import (
    get_tokenizer,
    chunk_text,
    convert_char_to_pinyin,
    list_str_to_idx,
)
import librosa
from maxdiffusion.utils.seq_utils import lens_to_mask
from flax import nnx
import flax.linen as nn
import flax
import jax_vocos


cfg_strength = 2




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

    #transformer_state = states

    loop_body_p = functools.partial(
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


def run(config):

    rng = jax.random.key(config.seed)
    rngs = nnx.Rngs(rng)
    devices_array = create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    # global_batch_size = config.per_device_batch_size * jax.local_device_count()

    # LOAD TRANSFORMER
    flash_block_sizes = get_flash_block_sizes(config)
    def get_f5_text_encoder():
        def create_model(rngs: nnx.Rngs, config: dict):
            f5_text_encoder = F5TextEmbedding(rngs=rngs,text_num_embeds=2545, text_dim=512, conv_layers=4)
            return f5_text_encoder

        p_model_factory = functools.partial(create_model, config=config)
        f5_text_encoder = nnx.eval_shape(p_model_factory, rngs=rngs)
        graphdef, state, rest_of_state = nnx.split(f5_text_encoder, nnx.Param, ...)
        # 3. retrieve the state shardings, mapping logical names to mesh axis names.
        logical_state_spec = nnx.get_partition_spec(state)
        logical_state_sharding = nn.logical_to_mesh_sharding(
            logical_state_spec, mesh, config.logical_axis_rules
        )
        logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
        params = state.to_pure_dict()
        state = dict(nnx.to_flat_state(state))

        params = load_f5_text_encoder(config.f5_text_encoder_pretrained_model_name_or_path, params, "cpu")
        params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)
        for path, val in flax.traverse_util.flatten_dict(params).items():
            # if restored_checkpoint:
            #     path = path[:-1]
            sharding = logical_state_sharding[path].value
            state[path].value = device_put_replicated(val, sharding)
        state = nnx.from_flat_state(state)
        f5_text_encoder = nnx.merge(graphdef, state, rest_of_state)
        return f5_text_encoder
    def get_f5_transformer():
        def create_model(rngs: nnx.Rngs, config: dict):
            f5_transformer = F5Transformer2DModel(
                text_dim=config.text_dim,  # Make sure text_dim is in config
                mel_dim=config.mel_dim,  # Make sure mel_dim is in config
                dim=config.latent_dim,  # Make sure latent_dim is in config
                head_dim=config.head_dim,
                num_depth=config.num_depth,
                num_heads=config.num_heads,
                mesh=mesh,
                attention_kernel=config.attention,
                flash_block_sizes=flash_block_sizes,
                dtype=config.activations_dtype,
                weights_dtype=config.weights_dtype,
                precision=get_precision(config),
                rngs=rngs,
            )
            return f5_transformer

        p_model_factory = functools.partial(create_model, config=config)
        f5_transformer = nnx.eval_shape(p_model_factory, rngs=rngs)
        graphdef, state, rest_of_state = nnx.split(f5_transformer, nnx.Param, ...)

        # 3. retrieve the state shardings, mapping logical names to mesh axis names.
        logical_state_spec = nnx.get_partition_spec(state)
        logical_state_sharding = nn.logical_to_mesh_sharding(
            logical_state_spec, mesh, config.logical_axis_rules
        )
        logical_state_sharding = dict(nnx.to_flat_state(logical_state_sharding))
        params = state.to_pure_dict()
        state = dict(nnx.to_flat_state(state))
        params = load_f5_transformer(config.f5_transformer_pretrained_model_name_or_path, params, "cpu",num_layers=config.num_depth)

        params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)
        for path, val in flax.traverse_util.flatten_dict(params).items():
            # if restored_checkpoint:
            #     path = path[:-1]
            sharding = logical_state_sharding[path].value
            state[path].value = device_put_replicated(val, sharding)
        state = nnx.from_flat_state(state)
        f5_transformer = nnx.merge(graphdef, state, rest_of_state)
        return f5_transformer
    # transformer_params = convert_f5_transformer_torch_to_nnx(config.pretrained_model_name_or_path)
    # weights_init_fn = functools.partial(transformer.init_weights, rngs=rng, max_sequence_length=config.max_sequence_length, eval_only=False)
    # transformer_state, transformer_state_shardings = setup_initial_state(
    #     model=transformer,
    #     tx=None,
    #     config=config,
    #     mesh=mesh,
    #     weights_init_fn=weights_init_fn,
    #     model_params=None,
    #     training=False,
    # )
    # transformer_state = transformer_state.replace(params=transformer_params)
    # transformer_state = jax.device_put(transformer_state, transformer_state_shardings)
    f5_text_encoder = get_f5_text_encoder()
    f5_transformer = get_f5_transformer()
    #get_memory_allocations()
    num_devices = len(jax.devices())
    data_sharding = jax.sharding.NamedSharding(mesh, P(*config.data_sharding))

    batch_size = 3 * num_devices
    local_speed = 1
    max_duration = 4096
    ref_text = "and there are so many things about humankind that is bad and evil. I strongly believe that love is one of the only things we have in this world."
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    gen_text = "Hello,I'm Aurora.And nice to meet you.This is a very long sentence intended to test the stability of the model.I really like this model and so I use it a lot."
    ref_audio, ref_sr = librosa.load("/home/fbs/jax-F5-TTS/test.mp3", sr=24000)
    max_chars = int(
        len(ref_text.encode("utf-8"))
        / (ref_audio.shape[-1] / ref_sr)
        * (22 - ref_audio.shape[-1] / ref_sr)
    )
    global_vocab_char_map, global_vocab_size = get_tokenizer(config.vocab_name_or_path, "custom")

    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    batched_text_list = []
    batched_duration = []
    ref_max_length = max_duration * 256
    ref_audio_len = ref_audio.shape[-1] // 256 + 1
    for single_gen_text in gen_text_batches:
        text_list = ref_text + single_gen_text
        ref_text_len = len(ref_text.encode("utf-8"))
        gen_text_len = len(single_gen_text.encode("utf-8"))
        duration = ref_audio_len + int(
            ref_audio_len / ref_text_len * gen_text_len / local_speed
        )
        batched_duration.append(duration)
        batched_text_list.append(text_list)

    final_text_list_pinyin = convert_char_to_pinyin(batched_text_list)
    global_max_sequence_length = config.max_sequence_length
    text_ids_unpadded,text_ids_mask = list_str_to_idx(final_text_list_pinyin, global_vocab_char_map, max_length=global_max_sequence_length)
    padded_batch_size = batch_size - len(gen_text_batches)
    text_ids = np.pad(text_ids_unpadded, ((0, padded_batch_size), (0, 0)))
    text_ids_mask = np.pad(text_ids_mask, ((0, padded_batch_size), (0, 0)))

    ref_audio = np.pad(ref_audio, (0, ref_max_length - 256 - ref_audio.shape[0]))

    # ref_audio = jax.device_put(
    #     ref_audio[np.newaxis, :], jax.sharding.NamedSharding(mesh, P(None, "data"))
    # )

    lens = np.full((batch_size,), ref_audio_len)
    duration = np.asarray(batched_duration)
    duration = np.pad(duration, (0, padded_batch_size))
    # duration = np.maximum(
    #     np.maximum((text_ids != 0).sum(axis=-1), lens) + 1, duration
    # )

    cond_mask = lens_to_mask(lens, length=config.max_sequence_length)
    mask = lens_to_mask(duration, length=config.max_sequence_length)

    cond = jax.jit(get_mel, out_shardings=None)(ref_audio)
    cond_mask = np.pad(
        cond_mask,
        ((0, batch_size - cond_mask.shape[0]), (0, max_duration - cond_mask.shape[-1])),
        constant_values=0,
    )
    mask = np.pad(
        mask,
        ((0, batch_size - mask.shape[0]), (0, max_duration - mask.shape[-1])),
        constant_values=0,
    )

    text_decoder_segment_ids = text_ids_mask.astype(np.int32)
    decoder_segment_ids = mask.astype(np.int32)

    step_cond = np.where(cond_mask[..., np.newaxis], cond, np.zeros_like(cond))

    latents = jax.random.normal(jax.random.PRNGKey(0), (batch_size, max_duration, 100))
    latents = jax.device_put(latents, data_sharding)
    step_cond = jax.device_put(step_cond, data_sharding)
    text_ids = jax.device_put(text_ids, data_sharding)

    t_start = 0
    timesteps = jnp.linspace(t_start, 1.0, config.num_inference_steps + 1).astype(
        jnp.float32
    )
    timesteps = timesteps + config.sway_sampling_coef * (
        jnp.cos(jnp.pi / 2 * timesteps) - 1 + timesteps
    )  # sway sampling
    c_ts = timesteps[:-1]
    p_ts = timesteps[1:]
    text_embed_cond = f5_text_encoder(
        text=text_ids,
        text_decoder_segment_ids=text_decoder_segment_ids,
    )
    text_embed_uncond = f5_text_encoder(
        text=jnp.zeros_like(text_ids),
        text_decoder_segment_ids=text_decoder_segment_ids,
    )
    # text_embed_cond = jitted_text_encode(
    #     {"params": text_encoder_params},
    #     text=text_ids,
    #     text_decoder_segment_ids=text_decoder_segment_ids,
    #     rngs=rng,
    # )
    # text_embed_uncond = jitted_text_encode(
    #     {"params": text_encoder_params},
    #     text=jnp.zeros_like(text_ids),
    #     text_decoder_segment_ids=text_decoder_segment_ids,
    #     rngs=rng,
    # )

    p_run_inference = functools.partial(
            run_inference,
            latents=latents,
            cond=step_cond,
            decoder_segment_ids=decoder_segment_ids,
            text_embed_cond=text_embed_cond,
            text_embed_uncond=text_embed_uncond,
            c_ts=c_ts,
            p_ts=p_ts,
        )
    graphdef, state, rest_of_state = nnx.split(f5_transformer, nnx.Param, ...)

    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        y_final = p_run_inference(graphdef=graphdef,
                                  sharded_state=state,
                                  rest_of_state=rest_of_state)
    out = y_final
    out = jnp.where(cond_mask[..., jnp.newaxis], cond, out)
    # from jax_vocos import load_model

    # vocos_model, vocos_params = load_model()
    vocos_vocoder = jax_vocos.load_model(load_path="/home/fbs/jax-test/vocos.safetensors")
    out = jax.device_put(out, data_sharding)
    # res = jax.jit(vocos_model.apply, out_shardings=None)(
    #     {"params": vocos_params}, out, rngs=rng
    # )
    res = vocos_vocoder(out)

    import soundfile as sf

    t0 = time.perf_counter()

    res_cpu = np.asarray(res)
    output_segment = res_cpu[0][ref_audio_len * 256 : duration[0] * 256]
    for i in range(batch_size - padded_batch_size):
        output_segment = np.concatenate(
            (
                output_segment,
                res_cpu[i + 1][ref_audio_len * 256 : duration[i + 1] * 256],
            )
        )
    sf.write("output.wav", output_segment, samplerate=24000)
    t1 = time.perf_counter()
    max_logging.log(f"transfer to cpu first and slice time: {t1 - t0:.1f}s.")

    return None


def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    run(pyconfig.config)


if __name__ == "__main__":
    app.run(main)
