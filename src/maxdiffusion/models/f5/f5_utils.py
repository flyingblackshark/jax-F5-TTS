import jax.numpy as jnp
from ..modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor, torch2jax, validate_flax_state_dict)
from safetensors import safe_open
from flax.traverse_util import unflatten_dict, flatten_dict
import jax
from maxdiffusion.common_types import F5_MODEL

def load_f5_transformer(
    pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True, num_layers: int = 40
):      
    with jax.default_device(device):
        tensors = {}
        with safe_open(
            #"/home/fbs/jax-test/aurora_f5_transformer.safetensors", 
            pretrained_model_name_or_path,
            framework="pt"
        ) as f:
            for k in f.keys():
                tensors[k] = torch2jax(f.get_tensor(k))
        flax_state_dict = {}
        cpu = jax.local_devices(backend="cpu")[0]
        flattened_dict = flatten_dict(eval_shapes)
        random_flax_state_dict = {}
        for key in flattened_dict:
            string_tuple = tuple([str(item) for item in key])
            random_flax_state_dict[string_tuple] = flattened_dict[key]
        del flattened_dict
        for pt_key, tensor in tensors.items():
            renamed_pt_key = rename_key(pt_key)
            renamed_pt_key = renamed_pt_key.replace("transformer.", "")
            renamed_pt_key = renamed_pt_key.replace(
                "transformer_blocks_", "transformer_blocks."
            )
            renamed_pt_key = renamed_pt_key.replace("conv1d_0.", "conv1.")
            renamed_pt_key = renamed_pt_key.replace("conv1d_2.", "conv2.")
            renamed_pt_key = renamed_pt_key.replace("time_mlp_", "time_mlp.layers.")
            renamed_pt_key = renamed_pt_key.replace("ff.ff_0.", "ff.layers.")
            renamed_pt_key = renamed_pt_key.replace("ff.ff_", "ff.layers.")
            # renamed_pt_key = renamed_pt_key.replace("ffn.net_2", "ffn.proj_out")
            # renamed_pt_key = renamed_pt_key.replace("ffn.net_0", "ffn.act_fn")
            # renamed_pt_key = renamed_pt_key.replace("norm2", "norm2.layer_norm")
            if "rotary_embed" in renamed_pt_key:
                continue
        
            pt_tuple_key = tuple(renamed_pt_key.split("."))

            if "transformer_blocks" in pt_tuple_key:
                new_key = ("transformer_blocks",) + pt_tuple_key[2:]
                block_index = int(pt_tuple_key[1])
                pt_tuple_key = new_key
            flax_key, flax_tensor = rename_key_and_reshape_tensor(
                pt_tuple_key, tensor, random_flax_state_dict,model_type=F5_MODEL
            )
            flax_key = rename_for_nnx(flax_key)
            flax_key = _tuple_str_to_int(flax_key)

            if "transformer_blocks" in flax_key:
                if flax_key in flax_state_dict:
                    new_tensor = flax_state_dict[flax_key]
                else:
                    new_tensor = jnp.zeros((num_layers,) + flax_tensor.shape) #config.num_depth
                flax_tensor = new_tensor.at[block_index].set(flax_tensor)
            flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
        # validate_flax_state_dict(eval_shapes, flax_state_dict)
        flax_state_dict = unflatten_dict(flax_state_dict)
        del tensors
        jax.clear_caches()
        return flax_state_dict
def load_f5_text_encoder(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True):
     with jax.default_device(device):
        tensors = {}
        with safe_open(
            #"/home/fbs/jax-test/aurora_f5_text_encoder.safetensors",
            pretrained_model_name_or_path,
             framework="pt"
        ) as f:
            for k in f.keys():
                tensors[k] = torch2jax(f.get_tensor(k))
        flax_state_dict = {}
        cpu = jax.local_devices(backend="cpu")[0]
        flattened_dict = flatten_dict(eval_shapes)
        random_flax_state_dict = {}
        for key in flattened_dict:
            string_tuple = tuple([str(item) for item in key])
            random_flax_state_dict[string_tuple] = flattened_dict[key]
        del flattened_dict
        for pt_key, tensor in tensors.items():
            renamed_pt_key = rename_key(pt_key)
            renamed_pt_key = renamed_pt_key.replace("transformer.text_embed.", "")
            renamed_pt_key = renamed_pt_key.replace(
                "text_blocks_", "text_blocks."
            )
            renamed_pt_key = renamed_pt_key.replace("text_embed.weight", "text_embed.embedding")
            renamed_pt_key = renamed_pt_key.replace("norm.", "layer_norm.")
            pt_tuple_key = tuple(renamed_pt_key.split("."))

            flax_key, flax_tensor = rename_key_and_reshape_tensor(
                pt_tuple_key, tensor, random_flax_state_dict,model_type=F5_MODEL
            )
            flax_key = rename_for_nnx(flax_key)
            flax_key = _tuple_str_to_int(flax_key)

            flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
        # validate_flax_state_dict(eval_shapes, flax_state_dict)
        flax_state_dict = unflatten_dict(flax_state_dict)
        del tensors
        jax.clear_caches()
        return flax_state_dict
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
    renamed_pt_key = renamed_pt_key.replace(
        "text_embedding_0", "condition_embedder.text_embedder.linear_1"
    )
    renamed_pt_key = renamed_pt_key.replace(
        "text_embedding_2", "condition_embedder.text_embedder.linear_2"
    )
    renamed_pt_key = renamed_pt_key.replace(
        "time_embedding_0", "condition_embedder.time_embedder.linear_1"
    )
    renamed_pt_key = renamed_pt_key.replace(
        "time_embedding_2", "condition_embedder.time_embedder.linear_2"
    )
    renamed_pt_key = renamed_pt_key.replace(
        "time_projection_1", "condition_embedder.time_proj"
    )

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