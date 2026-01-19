
import os
from absl import app
from typing import Sequence
import jax.numpy as jnp
from maxdiffusion import pyconfig

import tensorflow as tf
import jax

import flax
from maxdiffusion.utils.mel_util import get_mel
from maxdiffusion.checkpointing.f5_checkpointer import F5Checkpointer
import numpy as np
from flax import nnx
from maxdiffusion.utils.pinyin_utils import (
    get_tokenizer,
    prompt_clean,
    convert_char_to_pinyin,
    list_str_to_idx,
)

def bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))

def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_example(mel, txt_embed, decoder_segment_ids):
  mel = tf.io.serialize_tensor(mel)
  txt_embed = tf.io.serialize_tensor(txt_embed)
  decoder_segment_ids = tf.io.serialize_tensor(decoder_segment_ids)
  feature = {
      "mel": bytes_feature(mel),
      "txt_embed": bytes_feature(txt_embed),
      "decoder_segment_ids": bytes_feature(decoder_segment_ids),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))


def mock_mel():
    max_sequence_length = 2048
    mock_audio = jax.random.normal(jax.random.PRNGKey(0), (1, 24000 * 10))
    mel = get_mel(mock_audio)
    mel_len = mel.shape[1]
    
    decoder_segment_ids = np.zeros((max_sequence_length,), dtype=np.int32)
    decoder_segment_ids[:mel_len] = 1
    
    return jnp.pad(mel, ((0,0),(0, max_sequence_length - mel.shape[1]),(0,0)), 'constant'), decoder_segment_ids


def encode_txt(pipeline, text_ids, text_ids_mask):
    text_embed_cond = pipeline.text_encoder(
        text=text_ids,
        text_decoder_segment_ids=text_ids_mask.astype(np.int32),
    )
    return text_embed_cond


def encode_prompt(
    pipeline,
    prompt: str,
    max_sequence_length: int
):
    prompt = prompt_clean(prompt)
    pinyin_inputs = convert_char_to_pinyin([prompt])

    list_idx_tensors = [pipeline.global_vocab_char_map.get(c, 0) for c in pinyin_inputs[0]]
    text_ids = np.asarray(list_idx_tensors, dtype=np.int32)
    text_ids = text_ids + 1
    text_ids_mask = np.ones_like(text_ids)
    text_ids = np.pad(text_ids, ((0, max_sequence_length - text_ids.shape[0])), 'constant', constant_values=0) 
    text_ids_mask = np.pad(text_ids_mask, ((0, max_sequence_length - text_ids_mask.shape[0])), 'constant', constant_values=0) 

    #text_ids, text_ids_mask = list_str_to_idx(pinyin_inputs,pipeline.global_vocab_char_map, max_length=max_sequence_length)

    text_embed_cond = encode_txt(pipeline, text_ids[None], text_ids_mask[None])
    return text_embed_cond


def txt2prompt(pipeline, txt):

    txt_embed = encode_prompt(pipeline, txt, max_sequence_length=2048)
    return txt_embed


def mock_text(pipeline):
    mock_text = "abc123"
    txt_embed = txt2prompt(pipeline, mock_text)
    return txt_embed


def generate_dataset(config):

    checkpoint_loader = F5Checkpointer(config, "F5_CHECKPOINT")
    pipeline, _, _ = checkpoint_loader.load_checkpoint()

    tfrecords_dir = config.tfrecords_dir
    # if not os.path.exists(tfrecords_dir):
    #     os.makedirs(tfrecords_dir)

    tfrecord_num = 0
    no_records_per_shard = config.no_records_per_shard
    global_record_count = 0
    
    file_path = os.path.join(tfrecords_dir, "file_%.2i-%i.tfrecord" % (tfrecord_num, (global_record_count + no_records_per_shard)))
    writer = tf.io.TFRecordWriter(file_path)
    
    shard_record_count = 0

    for i in range(100):
        mel, decoder_segment_ids = mock_mel()
        txt_embed = mock_text(pipeline)
        # Write the example, including the timestep if applicable
        example = create_example(mel[0], txt_embed[0], decoder_segment_ids)
        writer.write(example.SerializeToString())
        shard_record_count += 1
        global_record_count += 1

        if shard_record_count >= no_records_per_shard:
            writer.close()
            tfrecord_num += 1
            file_path = os.path.join(tfrecords_dir, "file_%.2i-%i.tfrecord" % (tfrecord_num, (global_record_count + no_records_per_shard)))
            writer = tf.io.TFRecordWriter(file_path)
            shard_record_count = 0
    
    # Close any open writer
    try:
        writer.close()
    except Exception:
        pass


def run(config):
    generate_dataset(config)


def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    flax.config.update('flax_always_shard_variable', False)
    run(pyconfig.config)


if __name__ == "__main__":
    app.run(main)