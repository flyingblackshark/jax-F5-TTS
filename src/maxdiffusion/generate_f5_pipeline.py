# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence
import jax
import time
import os
from maxdiffusion.pipelines.f5.f5_pipeline import F5Pipeline
from maxdiffusion import pyconfig, max_logging, max_utils
from absl import app
from maxdiffusion.utils import export_to_video
from google.cloud import storage
import librosa
from maxdiffusion.utils.pinyin_utils import (
    get_tokenizer,
    chunk_text,
    convert_char_to_pinyin,
    list_str_to_idx,
)

jax.config.update("jax_use_shardy_partitioner", True)

def run(config, pipeline=None, filename_prefix=""):
  print("seed: ", config.seed)
  from maxdiffusion.checkpointing.f5_checkpointer import F5Checkpointer

  checkpoint_loader = F5Checkpointer(config, "F5_CHECKPOINT")
  pipeline = checkpoint_loader.load_checkpoint()
  # if pipeline is None:
  #   pipeline = F5Pipeline.from_pretrained(config)
  s0 = time.perf_counter()

  # Using global_batch_size_to_train_on so not to create more config variables
  ref_text = "and there are so many things about humankind that is bad and evil. I strongly believe that love is one of the only things we have in this world."
  gen_text = "Hello,I'm Aurora.And nice to meet you.This is a very long sentence intended to test the stability of the model.I really like this model and so I use it a lot."
  ref_audio, ref_sr = librosa.load("/home/fbs/jax-F5-TTS/test.mp3", sr=24000)
  local_speed = 1
  # max_logging.log(
  #     f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}"
  # )
  max_chars = int(
      len(ref_text.encode("utf-8"))
      / (ref_audio.shape[-1] / ref_sr)
      * (22 - ref_audio.shape[-1] / ref_sr)
  )
  gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
  ref_audio_len = ref_audio.shape[-1] // 256 + 1
  batched_text_list = []
  batched_duration = []
  for single_gen_text in gen_text_batches:
      text_list = ref_text + single_gen_text
      ref_text_len = len(ref_text.encode("utf-8"))
      gen_text_len = len(single_gen_text.encode("utf-8"))
      duration = ref_audio_len + int(
          ref_audio_len / ref_text_len * gen_text_len / local_speed
      )
      batched_duration.append(duration)
      batched_text_list.append(text_list)

  audios = pipeline(
      prompt=batched_text_list,
      reference_audio=["/home/fbs/jax-F5-TTS/test.mp3" for i in range(len(batched_text_list))],
      duration=duration,
      max_sequence_length=2048,
  )
  import soundfile as sf
  import numpy as np
  res_cpu = np.asarray(audios)
  output_segment = res_cpu[0][ref_audio_len * 256 : batched_duration[0] * 256]
  for i in range(len(batched_duration)-1):
      output_segment = np.concatenate(
          (
              output_segment,
              res_cpu[i + 1][ref_audio_len * 256 : batched_duration[i + 1] * 256],
          )
      )
  sf.write("output.wav", output_segment, samplerate=24000)
  print("compile time: ", (time.perf_counter() - s0))

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  run(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
