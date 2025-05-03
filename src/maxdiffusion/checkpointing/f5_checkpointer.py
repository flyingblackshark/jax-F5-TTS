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

from abc import ABC
from contextlib import nullcontext
import functools
import json
import jax
from jax.sharding import Mesh
import orbax.checkpoint as ocp
import grain.python as grain
from maxdiffusion import (
    max_utils,
    FlaxAutoencoderKL,
    max_logging,
)
from maxdiffusion.models.f5.transformers.transformer_f5_flax import F5Transformer2DModel,F5TextEmbedding
from ..pipelines.f5.f5_pipeline import F5Pipeline

#from transformers import (CLIPTokenizer, FlaxCLIPTextModel, FlaxT5EncoderModel, AutoTokenizer)

from maxdiffusion.checkpointing.checkpointing_utils import (create_orbax_checkpoint_manager)
#from maxdiffusion.models.f5.util import load_flow_model

F5_CHECKPOINT = "F5_CHECKPOINT"
_CHECKPOINT_FORMAT_ORBAX = "CHECKPOINT_FORMAT_ORBAX"

F5_STATE_KEY = "F5_state"
F5_TRANSFORMER_PARAMS_KEY = "F5_transformer_params"
F5_STATE_SHARDINGS_KEY = "F5_state_shardings"
F5_TEXT_ENCODER_KEY = "F5_text_encoder"
#F5_VAE_PARAMS_KEY = "F5_vae"
#VAE_STATE_KEY = "vae_state"
#VAE_STATE_SHARDINGS_KEY = "vae_state_shardings"


class F5Checkpointer(ABC):

  def __init__(self, config, checkpoint_type):
    self.config = config
    self.checkpoint_type = checkpoint_type
    self.checkpoint_format = None

    self.rng = jax.random.PRNGKey(self.config.seed)
    self.devices_array = max_utils.create_device_mesh(config)
    self.mesh = Mesh(self.devices_array, self.config.mesh_axes)
    self.total_train_batch_size = self.config.total_train_batch_size

    self.checkpoint_manager = create_orbax_checkpoint_manager(
        self.config.checkpoint_dir,
        enable_checkpointing=True,
        save_interval_steps=1,
        checkpoint_type=checkpoint_type,
        dataset_type=config.dataset_type,
    )

  def _create_optimizer(self, config, learning_rate):

    learning_rate_scheduler = max_utils.create_learning_rate_schedule(
        learning_rate, config.learning_rate_schedule_steps, config.warmup_steps_fraction, config.max_train_steps
    )
    tx = max_utils.create_optimizer(config, learning_rate_scheduler)
    return tx, learning_rate_scheduler

  def create_f5_state(self, pipeline, params, checkpoint_item_name, is_training):
    transformer = pipeline.f5

    tx, learning_rate_scheduler = None, None
    if is_training:
      learning_rate = self.config.learning_rate

      tx, learning_rate_scheduler = self._create_optimizer(self.config, learning_rate)

    transformer_eval_params = transformer.init_weights(
        rngs=self.rng, max_sequence_length=self.config.max_sequence_length, eval_only=True
    )
    transformer_params = transformer_eval_params
    #transformer_params = load_flow_model(self.config.F5_name, transformer_eval_params, "cpu")

    weights_init_fn = functools.partial(
        pipeline.f5.init_weights, rngs=self.rng, max_sequence_length=self.config.max_sequence_length
    )
    f5_state, state_mesh_shardings = max_utils.setup_initial_state(
        model=pipeline.f5,
        tx=tx,
        config=self.config,
        mesh=self.mesh,
        weights_init_fn=weights_init_fn,
        model_params=None,
        checkpoint_manager=self.checkpoint_manager,
        checkpoint_item=checkpoint_item_name,
        training=is_training,
    )
    if not self.config.train_new_f5:
      f5_state = f5_state.replace(params=transformer_params)
      f5_state = jax.device_put(f5_state, state_mesh_shardings)
    return f5_state, state_mesh_shardings, learning_rate_scheduler
  
  def restore_data_iterator_state(self, data_iterator):
    if (
        self.config.dataset_type == "grain"
        and data_iterator is not None
        and (self.checkpoint_manager.directory / str(self.checkpoint_manager.latest_step()) / "iter").exists()
    ):
      max_logging.log("Restoring data iterator from checkpoint")
      restored = self.checkpoint_manager.restore(
          self.checkpoint_manager.latest_step(),
          args=ocp.args.Composite(iter=grain.PyGrainCheckpointRestore(data_iterator.local_iterator)),
      )
      data_iterator.local_iterator = restored["iter"]
    else:
      max_logging.log("data iterator checkpoint not found")
    return data_iterator

  def _get_pipeline_class(self):
    return F5Pipeline

  def _set_checkpoint_format(self, checkpoint_format):
    self.checkpoint_format = checkpoint_format

  def save_checkpoint(self, train_step, pipeline, train_states):
    # def config_to_json(model_or_config):
    #   return json.loads(model_or_config.to_json_string())

    items = {
        #"f5_config": ocp.args.JsonSave(config_to_json(pipeline.f5)),
        #"vae_config": ocp.args.JsonSave(config_to_json(pipeline.vae)),
        #"scheduler_config": ocp.args.JsonSave(config_to_json(pipeline.scheduler)),
    }

    items[F5_STATE_KEY] = ocp.args.PyTreeSave(train_states[F5_STATE_KEY])
    #items["vae_state"] = ocp.args.PyTreeSave(train_states["vae_state"])
    #items["scheduler"] = ocp.args.PyTreeSave(train_states["scheduler"])

    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))

  def load_params(self, step=None):

    self.checkpoint_format = _CHECKPOINT_FORMAT_ORBAX

  def load_f5_configs_from_orbax(self, step):
    max_logging.log("Restoring stable diffusion configs")
    if step is None:
      step = self.checkpoint_manager.latest_step()
      if step is None:
        return None

    restore_args = {
        #"f5_config": ocp.args.JsonRestore(),
        #"vae_config": ocp.args.JsonRestore(),
        #"scheduler_config": ocp.args.JsonRestore(),
    }

    return (self.checkpoint_manager.restore(step, args=ocp.args.Composite(**restore_args)), None)

  def load_diffusers_checkpoint(self):
    flash_block_sizes = max_utils.get_flash_block_sizes(self.config)

    if jax.device_count() == jax.local_device_count():
      context = jax.default_device(jax.devices("cpu")[0])
    else:
      context = nullcontext()

    with context:
      # loading from pretrained here causes a crash when trying to compile the model
      # Failed to load HSACO: HIP_ERROR_NoBinaryForGpu
      transformer = F5Transformer2DModel(
          mesh=self.mesh,
          split_head_dim=self.config.split_head_dim,
          attention_kernel=self.config.attention,
          flash_block_sizes=flash_block_sizes,
          dtype=self.config.activations_dtype,
          weights_dtype=self.config.weights_dtype,
          precision=max_utils.get_precision(self.config),
      )
      transformer_eval_params = transformer.init_weights(
          rngs=self.rng, max_sequence_length=self.config.max_sequence_length, eval_only=True
      )
      transformer_params = transformer_eval_params

      text_encoder = F5TextEmbedding(
          mesh=self.mesh,
          split_head_dim=self.config.split_head_dim,
          attention_kernel=self.config.attention,
          flash_block_sizes=flash_block_sizes,
          dtype=self.config.activations_dtype,
          weights_dtype=self.config.weights_dtype,
          precision=max_utils.get_precision(self.config),
      )
      text_encoder_eval_params = text_encoder.init_weights(
          rngs=self.rng, max_sequence_length=self.config.max_sequence_length, eval_only=True
      )
      text_encoder_params = text_encoder_eval_params
      #transformer_params = load_flow_model(self.config.F5_name, transformer_eval_params, "cpu")

    pipeline = F5Pipeline(
        transformer,
        text_encoder,
        None,
        dtype=self.config.activations_dtype,
        mesh=self.mesh,
        config=self.config,
        rng=self.rng,
    )

    params = {F5_TEXT_ENCODER_KEY: text_encoder_params, F5_TRANSFORMER_PARAMS_KEY: transformer_params}

    return pipeline, params

  def load_checkpoint(self, step=None, scheduler_class=None):
    pipeline, params = None, {}
    if jax.device_count() == jax.local_device_count():
      context = jax.default_device(jax.devices("cpu")[0])
    else:
      context = nullcontext()

    with context:
      transformer = F5Transformer2DModel(
          mesh=self.mesh,
          split_head_dim=self.config.split_head_dim,
          attention_kernel=self.config.attention,
          flash_block_sizes=max_utils.get_flash_block_sizes(self.config),
          dtype=self.config.activations_dtype,
          weights_dtype=self.config.weights_dtype,
          precision=max_utils.get_precision(self.config),
          from_pt=self.config.from_pt,
      )
      text_encoder = F5TextEmbedding(
          mesh=self.mesh,
          split_head_dim=self.config.split_head_dim,
          attention_kernel=self.config.attention,
          flash_block_sizes=flash_block_sizes,
          dtype=self.config.activations_dtype,
          weights_dtype=self.config.weights_dtype,
          precision=max_utils.get_precision(self.config),
      )
      pipeline = F5Pipeline(
          transformer,
          text_encoder,
          None,
          dtype=self.config.activations_dtype,
          mesh=self.mesh,
          config=self.config,
          rng=self.rng,
      )

    return pipeline, params
