import torch
from ..modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor, torch2jax, validate_flax_state_dict)

def _tuple_str_to_int(in_tuple):
  out_list = []
  for item in in_tuple:
    try:
      out_list.append(int(item))
    except ValueError:
      out_list.append(item)
  return tuple(out_list)


def rename_for_nnx(key):
  new_key = key
  if "norm_k" in key or "norm_q" in key:
    new_key = key[:-1] + ("scale",)
  return new_key


def rename_for_custom_trasformer(key):
  renamed_pt_key = key.replace("model.diffusion_model.", "")

  renamed_pt_key = renamed_pt_key.replace("head.modulation", "scale_shift_table")
  renamed_pt_key = renamed_pt_key.replace("head.head", "proj_out")
  renamed_pt_key = renamed_pt_key.replace("text_embedding_0", "condition_embedder.text_embedder.linear_1")
  renamed_pt_key = renamed_pt_key.replace("text_embedding_2", "condition_embedder.text_embedder.linear_2")
  renamed_pt_key = renamed_pt_key.replace("time_embedding_0", "condition_embedder.time_embedder.linear_1")
  renamed_pt_key = renamed_pt_key.replace("time_embedding_2", "condition_embedder.time_embedder.linear_2")
  renamed_pt_key = renamed_pt_key.replace("time_projection_1", "condition_embedder.time_proj")

  renamed_pt_key = renamed_pt_key.replace("blocks_", "blocks.")
  renamed_pt_key = renamed_pt_key.replace("self_attn", "attn1")
  renamed_pt_key = renamed_pt_key.replace("cross_attn", "attn2")
  renamed_pt_key = renamed_pt_key.replace(".q.", ".query.")
  renamed_pt_key = renamed_pt_key.replace(".k.", ".key.")
  renamed_pt_key = renamed_pt_key.replace(".v.", ".value.")
  renamed_pt_key = renamed_pt_key.replace(".o.", ".proj_attn.")
  renamed_pt_key = renamed_pt_key.replace("ffn_0", "ffn.act_fn.proj")
  renamed_pt_key = renamed_pt_key.replace("ffn_2", "ffn.proj_out")
  renamed_pt_key = renamed_pt_key.replace(".modulation", ".scale_shift_table")
  renamed_pt_key = renamed_pt_key.replace("norm3", "norm2.layer_norm")

  return renamed_pt_key
# 参考代码
# def load_wan_vae(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True):
#   device = jax.devices(device)[0]
#   subfolder = "vae"
#   filename = "diffusion_pytorch_model.safetensors"
#   if os.path.isdir(pretrained_model_name_or_path):
#     ckpt_path = os.path.join(pretrained_model_name_or_path, subfolder, filename)
#     if not os.path.isfile(ckpt_path):
#       raise FileNotFoundError(f"File {ckpt_path} not found for local directory.")
#   elif hf_download:
#     ckpt_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename=filename)
#   max_logging.log(f"Load and port Wan 2.1 VAE on {device}")
#   with jax.default_device(device):
#     if ckpt_path is not None:
#       tensors = {}
#       with safe_open(ckpt_path, framework="pt") as f:
#         for k in f.keys():
#           tensors[k] = torch2jax(f.get_tensor(k))
#       flax_state_dict = {}
#       cpu = jax.local_devices(backend="cpu")[0]
#       for pt_key, tensor in tensors.items():
#         renamed_pt_key = rename_key(pt_key)
#         # Order matters
#         renamed_pt_key = renamed_pt_key.replace("up_blocks_", "up_blocks.")
#         renamed_pt_key = renamed_pt_key.replace("mid_block_", "mid_block.")
#         renamed_pt_key = renamed_pt_key.replace("down_blocks_", "down_blocks.")

#         renamed_pt_key = renamed_pt_key.replace("conv_in.bias", "conv_in.conv.bias")
#         renamed_pt_key = renamed_pt_key.replace("conv_in.weight", "conv_in.conv.weight")
#         renamed_pt_key = renamed_pt_key.replace("conv_out.bias", "conv_out.conv.bias")
#         renamed_pt_key = renamed_pt_key.replace("conv_out.weight", "conv_out.conv.weight")
#         renamed_pt_key = renamed_pt_key.replace("attentions_", "attentions.")
#         renamed_pt_key = renamed_pt_key.replace("resnets_", "resnets.")
#         renamed_pt_key = renamed_pt_key.replace("upsamplers_", "upsamplers.")
#         renamed_pt_key = renamed_pt_key.replace("resample_", "resample.")
#         renamed_pt_key = renamed_pt_key.replace("conv1.bias", "conv1.conv.bias")
#         renamed_pt_key = renamed_pt_key.replace("conv1.weight", "conv1.conv.weight")
#         renamed_pt_key = renamed_pt_key.replace("conv2.bias", "conv2.conv.bias")
#         renamed_pt_key = renamed_pt_key.replace("conv2.weight", "conv2.conv.weight")
#         renamed_pt_key = renamed_pt_key.replace("time_conv.bias", "time_conv.conv.bias")
#         renamed_pt_key = renamed_pt_key.replace("time_conv.weight", "time_conv.conv.weight")
#         renamed_pt_key = renamed_pt_key.replace("quant_conv", "quant_conv.conv")
#         renamed_pt_key = renamed_pt_key.replace("conv_shortcut", "conv_shortcut.conv")
#         if "decoder" in renamed_pt_key:
#           renamed_pt_key = renamed_pt_key.replace("resample.1.bias", "resample.layers.1.bias")
#           renamed_pt_key = renamed_pt_key.replace("resample.1.weight", "resample.layers.1.weight")
#         if "encoder" in renamed_pt_key:
#           renamed_pt_key = renamed_pt_key.replace("resample.1", "resample.conv")
#         pt_tuple_key = tuple(renamed_pt_key.split("."))
#         flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, eval_shapes)
#         flax_key = _tuple_str_to_int(flax_key)
#         flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
#       validate_flax_state_dict(eval_shapes, flax_state_dict)
#       flax_state_dict = unflatten_dict(flax_state_dict)
#       del tensors
#       jax.clear_caches()
#     else:
#       raise FileNotFoundError(f"Path {ckpt_path} was not found")

#     return flax_state_dict
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