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

"""
Prepare tfrecords with latents and text embeddings preprocessed.
1. Download the dataset
"""

import os
from absl import app
from typing import Sequence
import csv
import jax.numpy as jnp
from maxdiffusion import pyconfig

import torch
import tensorflow as tf
import pickle
from array_record.python import array_record_module
import jax

import flax
from maxdiffusion.utils.mel_util import get_mel
from maxdiffusion.checkpointing.f5_checkpointer import F5Checkpointer
import numpy as np
from flax import nnx
from maxdiffusion.utils.pinyin_utils import (
    get_tokenizer,
    chunk_text,
    convert_char_to_pinyin,
    list_str_to_idx,
    prompt_clean,
)



def create_example(mel, txt_embed):
  feature = {
      "mel": mel,
      "txt_embed": txt_embed,
  }

  return pickle.dumps(feature)



def mock_mel():
  max_sequence_length = 2048
  mock_audio = jax.random.normal(jax.random.PRNGKey(0), (1, 24000 * 10))
  mel = np.asarray(get_mel(mock_audio))
  return np.pad(mel,(0, max_sequence_length - mel.shape[0]),'constant')

def encode_txt(pipeline,text_ids,text_ids_mask):
  text_embed_cond  = pipeline.text_encoder(
    text = text_ids,
    text_decoder_segment_ids=text_ids_mask.astype(np.int32),
  )

def encode_prompt(
    pipeline,
    prompt: str,
    max_sequence_length:int,
    global_vocab_char_map
):
  prompt = prompt_clean(prompt)
  pinyin_inputs = convert_char_to_pinyin(prompt)
  
  text_ids,text_ids_mask = list_str_to_idx(pinyin_inputs, global_vocab_char_map, max_length=max_sequence_length)

  text_embed_cond = encode_txt(pipeline,text_ids,text_ids_mask)
  return text_embed_cond
def txt2prompt(pipeline,txt,global_vocab_char_map):

  txt_embed = encode_prompt(pipeline,txt,max_sequence_length=2048,global_vocab_char_map=global_vocab_char_map)
  return np.asarray(txt_embed)

def mock_text(pipeline,global_vocab_char_map):
  mock_text = "abc123"
  txt_embed = txt2prompt(pipeline,mock_text,global_vocab_char_map)
  return txt_embed

def generate_dataset(config):
  global_vocab_char_map, _ = get_tokenizer(config.vocab_name_or_path, "custom")
  checkpoint_loader = F5Checkpointer(config, "F5_CHECKPOINT")
  pipeline,_,_ = checkpoint_loader.load_checkpoint()

  grainrecords_dir = config.grainrecords_dir
  if not os.path.exists(grainrecords_dir):
    os.makedirs(grainrecords_dir)

  grain_rec_num = 0
  no_records_per_shard = config.no_records_per_shard
  global_record_count = 0
  writer = array_record_module.ArrayRecordWriter(
      grainrecords_dir + "/file_%.2i-%i.array_record" % (grain_rec_num, (global_record_count + no_records_per_shard)),
      "group_size:1"
  )
  shard_record_count = 0


  for i in range(100):
    mel = mock_mel()
    txt_embed = mock_text(pipeline,global_vocab_char_map)
    # Write the example, including the timestep if applicable
    writer.write(create_example(mel, txt_embed))
    shard_record_count += 1
    global_record_count += 1

    if shard_record_count >= no_records_per_shard:
      writer.close()
      grain_rec_num += 1
      writer = array_record_module.ArrayRecordWriter(
          grainrecords_dir + "/file_%.2i-%i.array_record" % (grain_rec_num, (global_record_count + no_records_per_shard)),
          "group_size:1"
      )
      shard_record_count = 0


def run(config):
  generate_dataset(config)


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  flax.config.update('flax_always_shard_variable', False)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)