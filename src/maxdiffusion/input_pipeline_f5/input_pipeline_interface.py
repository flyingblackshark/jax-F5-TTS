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

import os
from functools import partial
import tensorflow as tf
from datasets import load_from_disk, Dataset
import jax

from maxdiffusion.input_pipeline_f5 import _hf_data_processing
# from maxdiffusion.input_pipeline_f5 import _grain_data_processing
# from maxdiffusion.input_pipeline_f5 import _tfds_data_processing
from maxdiffusion import multihost_dataloading
#from maxdiffusion.maxdiffusion_utils import tokenize_captions, transform_images, vae_apply
from maxdiffusion.dreambooth.dreambooth_constants import (
    INSTANCE_IMAGES,
    INSTANCE_IMAGE_LATENTS,
    INSTANCE_PROMPT_IDS,
    INSTANCE_PROMPT_INPUT_IDS,
    CLASS_IMAGES,
    CLASS_IMAGE_LATENTS,
    CLASS_PROMPT_IDS,
    CLASS_PROMPT_INPUT_IDS,
    INSTANCE_DATASET_NAME,
    CLASS_DATASET_NAME,
)
from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE


def make_data_iterator(
    config,
    dataloading_host_index,
    dataloading_host_count,
    mesh,
    global_batch_size,
    tokenize_fn=None,
    image_transforms_fn=None,
):
  """Make data iterator for SD1, 2, XL, dataset_types in (hf, tf, tfrecord)"""
  if config.dataset_type == "hf":
    return _hf_data_processing.make_hf_streaming_iterator(
        config,
        dataloading_host_index,
        dataloading_host_count,
        mesh,
        global_batch_size,
        tokenize_fn=tokenize_fn,
        image_transforms_fn=image_transforms_fn,
    )
  # elif config.dataset_type == "grain":
  #   return _grain_data_processing.make_grain_iterator(
  #       config,
  #       dataloading_host_index,
  #       dataloading_host_count,
  #       mesh,
  #       global_batch_size,
  #   )
  # elif config.dataset_type == "tf":
  #   return _tfds_data_processing.make_tf_iterator(
  #       config,
  #       dataloading_host_index,
  #       dataloading_host_count,
  #       mesh,
  #       global_batch_size,
  #       tokenize_fn=tokenize_fn,
  #       image_transforms_fn=image_transforms_fn,
  #   )
  # elif config.dataset_type == "tfrecord":
  #   return _tfds_data_processing.make_tfrecord_iterator(
  #       config,
  #       dataloading_host_index,
  #       dataloading_host_count,
  #       mesh,
  #       global_batch_size,
  #   )
  else:
    assert False, f"Unknown dataset_type {config.dataset_type}, dataset_type must be in (tf, tfrecord, hf, grain)"