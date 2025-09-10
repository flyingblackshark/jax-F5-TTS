import torch
from ..modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor, torch2jax, validate_flax_state_dict)
from safetensors import safe_open
from flax.traverse_util import unflatten_dict, flatten_dict
import jax


def convert_f5_transformer_torch_to_nnx(ckpt_path):
    tensors = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = torch2jax(f.get_tensor(k))
            
    state_dict = state_dict["model_state_dict"]
    params = {}

    params[f"time_embed.linear1.kernel"] = state_dict[f"transformer.time_embed.time_mlp.0.weight"].T
    params[f"time_embed.linear1.bias"] = state_dict[f"transformer.time_embed.time_mlp.0.bias"]
    params[f"time_embed.linear2.kernel"] = state_dict[f"transformer.time_embed.time_mlp.2.weight"].T
    params[f"time_embed.linear2.bias"] = state_dict[f"transformer.time_embed.time_mlp.2.bias"]

    params[f"input_embed.Dense_0.kernel"] = state_dict[f"transformer.input_embed.proj.weight"].T
    params[f"input_embed.Dense_0.bias"] = state_dict[f"transformer.input_embed.proj.bias"]
    params[f"input_embed.ConvPositionEmbedding_0.Conv_0.kernel"] = state_dict[f"transformer.input_embed.conv_pos_embed.conv1d.0.weight"].transpose(0,2)
    params[f"input_embed.ConvPositionEmbedding_0.Conv_0.bias"] = state_dict[f"transformer.input_embed.conv_pos_embed.conv1d.0.bias"]
    params[f"input_embed.ConvPositionEmbedding_0.Conv_1.kernel"] = state_dict[f"transformer.input_embed.conv_pos_embed.conv1d.2.weight"].transpose(0,2)
    params[f"input_embed.ConvPositionEmbedding_0.Conv_1.bias"] = state_dict[f"transformer.input_embed.conv_pos_embed.conv1d.2.bias"]



    for i in range(22):
        params[f"blocks_{i}.attn.to_k.kernel"] = state_dict[f"transformer.transformer_blocks.{i}.attn.to_k.weight"].T
        params[f"blocks_{i}.attn.to_k.bias"] = state_dict[f"transformer.transformer_blocks.{i}.attn.to_k.bias"]
        params[f"blocks_{i}.attn.to_q.kernel"] = state_dict[f"transformer.transformer_blocks.{i}.attn.to_q.weight"].T
        params[f"blocks_{i}.attn.to_q.bias"] = state_dict[f"transformer.transformer_blocks.{i}.attn.to_q.bias"]
        params[f"blocks_{i}.attn.to_v.kernel"] = state_dict[f"transformer.transformer_blocks.{i}.attn.to_v.weight"].T
        params[f"blocks_{i}.attn.to_v.bias"] = state_dict[f"transformer.transformer_blocks.{i}.attn.to_v.bias"]
        params[f"blocks_{i}.attn.to_out_0.kernel"] = state_dict[f"transformer.transformer_blocks.{i}.attn.to_out.0.weight"].T
        params[f"blocks_{i}.attn.to_out_0.bias"] = state_dict[f"transformer.transformer_blocks.{i}.attn.to_out.0.bias"]
        params[f"blocks_{i}.attn_norm.lin.kernel"] = state_dict[f"transformer.transformer_blocks.{i}.attn_norm.linear.weight"].T
        params[f"blocks_{i}.attn_norm.lin.bias"] = state_dict[f"transformer.transformer_blocks.{i}.attn_norm.linear.bias"]
        params[f"blocks_{i}.ff.layers_0.kernel"] = state_dict[f"transformer.transformer_blocks.{i}.ff.ff.0.0.weight"].T
        params[f"blocks_{i}.ff.layers_0.bias"] = state_dict[f"transformer.transformer_blocks.{i}.ff.ff.0.0.bias"]
        params[f"blocks_{i}.ff.layers_2.kernel"] = state_dict[f"transformer.transformer_blocks.{i}.ff.ff.2.weight"].T
        params[f"blocks_{i}.ff.layers_2.bias"] = state_dict[f"transformer.transformer_blocks.{i}.ff.ff.2.bias"]


    params[f"proj_out.kernel"] = state_dict[f"transformer.proj_out.weight"].T
    params[f"proj_out.bias"] = state_dict[f"transformer.proj_out.bias"]
    
    params[f"norm_out.Dense_0.kernel"] = state_dict[f"transformer.norm_out.linear.weight"].T
    params[f"norm_out.Dense_0.bias"] = state_dict[f"transformer.norm_out.linear.bias"]
    params = {k: v.cpu().numpy() for k, v in params.items()}
    params = unflatten_dict(params, sep=".")

    return params


def convert_f5_transformer_torch_to_nnx(ckpt_path):
    tensors = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = torch2jax(f.get_tensor(k))
    flax_state_dict = {}
    cpu = jax.local_devices(backend="cpu")[0]
    flattened_dict = flatten_dict(eval_shapes)

    state_dict = state_dict["model_state_dict"]
    params = {}
    text_encoder_params = {}

    text_encoder_params["text_embed.embedding"] = state_dict["transformer.text_embed.text_embed.weight"]
    for i in range(4):
        text_encoder_params[f"text_blocks_{i}.Conv_0.kernel"] = state_dict[f"transformer.text_embed.text_blocks.{i}.dwconv.weight"].transpose(0,2)
        text_encoder_params[f"text_blocks_{i}.Conv_0.bias"] = state_dict[f"transformer.text_embed.text_blocks.{i}.dwconv.bias"]
        text_encoder_params[f"text_blocks_{i}.GRN_0.gamma"] = state_dict[f"transformer.text_embed.text_blocks.{i}.grn.gamma"]
        text_encoder_params[f"text_blocks_{i}.GRN_0.beta"] = state_dict[f"transformer.text_embed.text_blocks.{i}.grn.beta"]
        text_encoder_params[f"text_blocks_{i}.LayerNorm_0.scale"] = state_dict[f"transformer.text_embed.text_blocks.{i}.norm.weight"].T
        text_encoder_params[f"text_blocks_{i}.LayerNorm_0.bias"] = state_dict[f"transformer.text_embed.text_blocks.{i}.norm.bias"]
        text_encoder_params[f"text_blocks_{i}.Dense_0.kernel"] = state_dict[f"transformer.text_embed.text_blocks.{i}.pwconv1.weight"].T
        text_encoder_params[f"text_blocks_{i}.Dense_0.bias"] = state_dict[f"transformer.text_embed.text_blocks.{i}.pwconv1.bias"]
        text_encoder_params[f"text_blocks_{i}.Dense_1.kernel"] = state_dict[f"transformer.text_embed.text_blocks.{i}.pwconv2.weight"].T
        text_encoder_params[f"text_blocks_{i}.Dense_1.bias"] = state_dict[f"transformer.text_embed.text_blocks.{i}.pwconv2.bias"]


    text_encoder_params = {k: v.cpu().numpy() for k, v in text_encoder_params.items()}
    text_encoder_params = unflatten_dict(text_encoder_params, sep=".")

    return params,text_encoder_params