import jax.numpy as jnp
from ..modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor, torch2jax, validate_flax_state_dict)
from safetensors import safe_open
from flax.traverse_util import unflatten_dict, flatten_dict
import jax
from maxdiffusion.models.modeling_flax_pytorch_utils import (rename_key, rename_key_and_reshape_tensor)

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
    
def load_vocos(pretrained_model_name_or_path: str, eval_shapes: dict, device: str, hf_download: bool = True):
     with jax.default_device(device):
        tensors = {}
        with safe_open(
            #"/home/fbs/jax-test/vocos.safetensors",
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
            # renamed_pt_key = renamed_pt_key.replace("transformer.text_embed.", "")
            # renamed_pt_key = renamed_pt_key.replace(
            #     "text_blocks_", "text_blocks."
            # )
            renamed_pt_key = renamed_pt_key.replace("convnext_", "convnext.")
            # renamed_pt_key = renamed_pt_key.replace("norm.", "layer_norm.")
            if "feature_extractor" in renamed_pt_key:
                continue
            if "head.istft" in renamed_pt_key:
                continue

            pt_tuple_key = tuple(renamed_pt_key.split("."))

            flax_key, flax_tensor = rename_key_and_reshape_tensor(
                pt_tuple_key, tensor, random_flax_state_dict
            )
            flax_key = rename_for_nnx(flax_key)
            flax_key = _tuple_str_to_int(flax_key)

            flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
        # validate_flax_state_dict(eval_shapes, flax_state_dict)
        flax_state_dict = unflatten_dict(flax_state_dict)
        del tensors
        jax.clear_caches()
        return flax_state_dict