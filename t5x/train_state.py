# Copyright 2022 The T5X Authors.
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

"""Train state for passing around objects during training."""

from typing import Any, Mapping, MutableMapping, Optional, Tuple

from flax import traverse_util
import flax.core
from flax.core import scope as flax_scope
from flax.linen import partitioning as flax_partitioning
import flax.serialization
import flax.struct
import jax.numpy as jnp
from t5x import optimizers

import jax
import optax
import functools
import gin

import tensorflow as tf
import os
import numpy as np
import pickle
from melodi.colabs.shared_code import datasets

import typing_extensions

EMPTY_DICT = flax.core.freeze({})
FrozenDict = flax_scope.FrozenDict
FrozenVariableDict = flax_scope.FrozenVariableDict
MutableVariableDict = flax_scope.MutableVariableDict
VariableDict = flax_scope.VariableDict


#  @functools.partial(jax.jit, backend='cpu')
@gin.configurable
def get_optax_optimizer(optimizer_name=None, melodi_path=None, learning_rate=0.3, momentum=0.0, melodi_memory=256, melodi_model='gradient'):

    if optimizer_name is None:
        optimizer_name = 'adafactor'
    if momentum == 0.0:
        momentum = None
    if melodi_path is None:
        melodi_path = 'gs://melodi-bucket0/melodi_training/task=glue_mnli_and_dev_v002/horizon=32/memory=256/bsz=64/lr=5e-5'

    import jax
    import optax
    import numpy as np

    from melodi.colabs.shared_code import optimizers
    from melodi.colabs.shared_code import models

    from flaxformer.architectures.t5 import t5_common_layers

    #  if optimizer is None:
        #  return optax.sgd(
            #  learning_rate=0.05,
        #  )
    #  return optax.adafactor(
        #  learning_rate=0.05,
        #  min_dim_size_to_factor=128,
        #  decay_rate=0.8,
        #  decay_offset=-1000000,
        #  multiply_by_parameter_scale=False,
        #  clipping_threshold=1.0,
        #  momentum=None,
        #  weight_decay_rate=1e-5,
        #  eps=1e-30,
        #  factored=True,
    #  )

    # MELODI DEFINITION:

    N_FEATURES = 2048
    prompt = np.random.randn(1, N_FEATURES)

    # instantiate optimizer
    embedder = models.NoOpEmbedder(
        num_embeddings=1,
        features=N_FEATURES,
        one_hot=True,
        name='token_embedder',
    )
    transformer = t5_common_layers.decoder(
        num_heads=6,
        head_dim=64,
        mlp_dim=1024,
        num_layers=8,
        shared_token_embedder=embedder,
        dropout_rate=0.0,
        activations=('gelu', 'linear'),
    )
    transformer = jax.tree_util.tree_map(lambda x: jax.device_get(x), transformer)
    if melodi_model == 'gradients':
        optimizer = optimizers.GradientOptimizer(
            model=optimizers.DecoderOnlyOptimizer(model=transformer),
        )
    elif melodi_model == 'multi_timescale':
        optimizer = optimizers.GradientMultiTimescaleOptimizer(
            model=optimizers.SequenceModelDecoderOnlyOptimizer(model=transformer),
            add_segment_embeddings=True,
            add_separator_embeddings=False,
            include_prompt_id=True,
        )

    #  if False: # random parameters
        #  rng = jax.random.PRNGKey(1234)
        #  parameters = optimizer.init(rng, {'gradients': prompt})
        #  preprocessor = None

    checkpoint_path = os.path.join(
        #  'gs://melodi-bucket0/melodi_training/horizon=32/memory=256/bsz=64/lr=5e-5',
        #  'gs://melodi-bucket0/melodi_training/task=squad_v010_allanswers/horizon=32/memory=256/bsz=64/lr=5e-5',
        melodi_path,
        'final_checkpoint.pkl',
    )
    with tf.io.gfile.GFile(name=checkpoint_path, mode='rb') as f:
        checkpoint = pickle.loads(f.read())

    # get parameters
    parameters = checkpoint['params']

    # get preprocessor
    preprocessor = datasets.DatasetPreprocessor()
    preprocessor.set_state(checkpoint['preprocessor'])

    melodi_optimizer = optimizers.PerTokenOptaxWrapper(
        optimizer,
        parameters,
        memory=melodi_memory,
        preprocessor=preprocessor,
    )

    # ADAFACTOR DEFINITION:

    adafactor = optax.adafactor(
        learning_rate=learning_rate,
        min_dim_size_to_factor=128,
        decay_rate=0.8,
        decay_offset=-1100000,
        multiply_by_parameter_scale=False,
        clipping_threshold=1.0,
        momentum=momentum,
        weight_decay_rate=1e-5,
        eps=1e-30,
        factored=True,
    )

    # HEAVYBALL DEFINITION

    heavyball = optax.sgd(
        learning_rate=learning_rate,
        momentum=momentum,
    )

    @jax.tree_util.register_pytree_node_class
    class ChainedOptimizer:

        def __init__(self, optimizers):
            self.optimizers = optimizers

        def init(self, prompt):
            state = []
            for opt in self.optimizers:
                state.append(opt.init(prompt))
            return state

        def update(self, gradients, states, prompt):
            update = gradients
            new_states = []
            for opt, state in zip(self.optimizers, states):
                update, new_state = opt.update(update, state, prompt)
                new_states.append(new_state)
            return update, new_states

        def tree_flatten(self):
            contents = []
            auxiliaries = [self.optimizers, ]
            return contents, auxiliaries

        @classmethod
        def tree_unflatten(self, auxiliaries, contents):
            return ChainedOptimizer(auxiliaries[0])

    @jax.tree_util.register_pytree_node_class
    class EnsembleOptimizer:

        def __init__(self, optimizers):
            self.optimizers = optimizers
            self.num_optimizers = len(optimizers)

        def init(self, prompt):
            state = []
            for opt in self.optimizers:
                state.append(opt.init(prompt))
            return state

        def update(self, gradients, states, prompt):
            update = jax.tree_util.tree_map(lambda g: 0.0, gradients)
            new_states = []
            for opt, state in zip(self.optimizers, states):
                opt_update, new_state = opt.update(gradients, state, prompt)
                update = jax.tree_util.tree_map(
                    lambda u, g: u + g / self.num_optimizers,
                    update,
                    opt_update,
                )
                new_states.append(new_state)
            return update, new_states

        def tree_flatten(self):
            contents = []
            auxiliaries = [self.optimizers, ]
            return contents, auxiliaries

        @classmethod
        def tree_unflatten(self, auxiliaries, contents):
            return ChainedOptimizer(auxiliaries[0])

    # NOISY GRADIENTS DEFINITION

    @jax.tree_util.register_pytree_node_class
    class NoisyGradients:

        def __init__(self, noise):
            self.noise = noise

        def init(self, prompt):
            state = []
            return state

        def update(self, gradients, states, prompt):
            update = jax.tree_util.tree_map(
                lambda x: x + self.noise * np.random.randn(*x.shape),
                gradients,
            )
            new_states = []
            return update, new_states

        def tree_flatten(self):
            contents = []
            auxiliaries = [self.noise, ]
            return contents, auxiliaries

        @classmethod
        def tree_unflatten(self, auxiliaries, contents):
            return NoisyGradients(auxiliaries[0])


    if optimizer_name == 'melodi':
        return ChainedOptimizer((melodi_optimizer, heavyball))
    elif optimizer_name == 'heavyball':
        return heavyball
    elif optimizer_name == 'adafactor':
        return adafactor
    elif optimizer_name == 'adafactor-noisy':
        return ChainedOptimizer((NoisyGradients(8e-4), adafactor))
    elif optimizer_name == 'adafactor+melodi':
        return EnsembleOptimizer((melodi_optimizer, adafactor))
    elif optimizer_name == 'adafactor+adafactor':
        return EnsembleOptimizer((adafactor, adafactor))
    elif optimizer_name == 'melofactor':
        return ChainedOptimizer((melodi_optimizer, adafactor))
    raise ValueError('Unknown optimizer =' + optimizer_name)

# Has to be on CPU when call from host_callback, else deadlocks
# when allocating new tensors.
@functools.partial(jax.jit, backend='cpu', static_argnames=['optimizer'])
def optax_init(prompt, optimizer):
    return jax.device_get(optimizer.init(prompt))


# Has to be on CPU when call from host_callback, else deadlocks
# when allocating new tensors.
@functools.partial(jax.jit, backend='cpu', static_argnames=['optimizer'])
def optax_update(prompt, grads, state, optimizer):
    update, new_state = optimizer.update(
        grads,
        state,
        prompt,
    )
    new_prompt = optax.apply_updates(prompt, update)
    return new_prompt, new_state


OPTAX_OPTIMIZER = None
OPTAX_STATE = None


@typing_extensions.runtime_checkable
class TrainState(typing_extensions.Protocol):
  """TrainState interface."""

  @property
  def step(self) -> jnp.ndarray:
    """The current training step as an integer scalar."""
    ...

  @property
  def params(self) -> FrozenVariableDict:
    """The parameters of the model as a PyTree matching the Flax module."""
    ...

  @property
  def param_states(self) -> FrozenVariableDict:
    """The optimizer states of the parameters as a PyTree."""
    ...

  @property
  def flax_mutables(self) -> FrozenVariableDict:
    """Flax mutable collection."""
    ...

  def state_dict(self) -> MutableVariableDict:
    """Returns a mutable representation of the state for checkpointing."""
    ...

  def restore_state(self, state_dict: Mapping[str, Any]) -> 'TrainState':
    """Restores the object state from a state dict."""
    ...

  def replace_params(self, params: VariableDict) -> 'TrainState':
    ...

  def replace_flax_mutables(self, flax_mutables: FrozenDict) -> 'TrainState':
    ...

  def replace_step(self, step: jnp.ndarray) -> 'TrainState':
    ...

  def apply_gradient(self,
                     grads,
                     learning_rate,
                     flax_mutables=EMPTY_DICT) -> 'TrainState':
    """Applies gradient, increments step, and returns an updated TrainState."""
    ...

  def as_logical_axes(self) -> 'TrainState':
    """Replaces `param` and `param-states` with their logical axis names."""
    ...


def _validate_params_axes(params_axes, params):
  axis_names = flax_partitioning.get_axis_names(params_axes)
  missing_params_axes = (
      set(traverse_util.flatten_dict(params, sep='/')) -
      set(traverse_util.flatten_dict(axis_names, sep='/')))
  if missing_params_axes:
    raise ValueError(
        f'Missing axis names for parameters: {missing_params_axes}')


def _split_variables_and_axes(
    variables_and_axes: FrozenVariableDict
) -> Tuple[FrozenVariableDict, FrozenVariableDict]:
  """Splits `variables_and_axes` into two separate dicts with the same keys."""
  # For each `key`, `key_axes` (if any) are its axes in `variables_and_axes`.
  variables = {}
  axes = {}
  for k, v in variables_and_axes.items():
    if k.endswith('_axes'):
      axes[k[:-5]] = v  # k without "_axes".
      _validate_params_axes(v, variables_and_axes[k[:-5]])  # k without "_axes".
    else:
      variables[k] = v
  return flax.core.freeze(variables), flax.core.freeze(axes)


class FlaxOptimTrainState(flax.struct.PyTreeNode):
  """Simple train state for holding parameters, step, optimizer state."""
  _optimizer: optimizers.OptimizerType
  # Contains axis metadata (e.g., names) matching parameter tree.
  params_axes: Optional[FrozenVariableDict] = None
  # Flax mutable fields.
  flax_mutables: FrozenDict = EMPTY_DICT
  # Contains axis metadata (e.g., names) matching flax_mutables tree.
  flax_mutables_axes: Optional[FrozenVariableDict] = None

  # optax stuff
  #  optax_optimizer: optax.GradientTransformation = None
  optax_state: tuple = None

  @classmethod
  def create(cls, optimizer_def: optimizers.OptimizerDefType,
             model_variables: FrozenVariableDict) -> 'FlaxOptimTrainState':
    other_variables, params = model_variables.pop('params')
    if 'params_axes' in other_variables:
      other_variables, params_axes = other_variables.pop('params_axes')
      _validate_params_axes(params_axes, params)
    else:
      params_axes = None

    # Split other_variables into mutables and their corresponding axes.
    flax_mutables, flax_mutables_axes = _split_variables_and_axes(
        other_variables)

    # If the optimizer supports `set_param_axes`, then assume that the model
    # code is emitting these axes and use it.
    if hasattr(optimizer_def, 'set_param_axes'):
      if params_axes is None:
        raise ValueError('The optimizer supports params_axes for model-based '
                         'partitioning, but the model is not emitting them.')
      # `get_axis_names` removes "_axes" suffix in the leaf name and replaces
      # `AxisMetadata` with `PartitionSpec`.
      axis_names = flax_partitioning.get_axis_names(params_axes)
      optimizer_def.set_param_axes(axis_names)

    optimizer = optimizer_def.create(params)
    flax_mutables_axes = flax_mutables_axes or None
    #  optax_optimizer = optax.chain(
        #  optax.clip_by_global_norm(1.0),
        #  optax.sgd(0.001, momentum=0.99),
    #  )
    #  optax_state = optax_optimizer.init(model_variables['params']['encoder']['prompt'])
    return FlaxOptimTrainState(
        optimizer,
        params_axes=params_axes,
        flax_mutables=flax_mutables,
        flax_mutables_axes=flax_mutables_axes,
        #  optax_optimizer=optax_optimizer,
        #  optax_state=optax_state,
    )

  @property
  def step(self) -> jnp.ndarray:
    return self._optimizer.state.step

  @property
  def params(self) -> FrozenVariableDict:
    return self._optimizer.target

  @property
  def param_states(self) -> FrozenVariableDict:
    return self._optimizer.state.param_states

  def state_dict(self) -> MutableVariableDict:
    state_dict = self._optimizer.state_dict()
    if self.flax_mutables:
      state_dict['flax_mutables'] = flax.core.unfreeze(self.flax_mutables)
    return state_dict

  def apply_gradient(self,
                     grads,
                     learning_rate,
                     flax_mutables=EMPTY_DICT) -> 'FlaxOptimTrainState':

    import jax.experimental.host_callback as hcb

    def local_update(*args):

        global OPTAX_STATE
        global OPTAX_OPTIMIZER

        if OPTAX_OPTIMIZER is None:
            OPTAX_OPTIMIZER = get_optax_optimizer()

        # unpack
        prompt = jax.device_get(args[0][0])['prompt']
        grads = jax.device_get(args[0][1])['prompt']

        if OPTAX_STATE is None:
            OPTAX_STATE = optax_init(prompt, optimizer=OPTAX_OPTIMIZER)
        state = jax.device_get(OPTAX_STATE)

        # Compute the update
        new_prompt, state = optax_update(
            prompt=prompt,
            grads=grads,
            state=state,
            optimizer=OPTAX_OPTIMIZER,
        )
        new_prompt = flax.core.freeze({'prompt': new_prompt})

        # update state and return new prompt
        OPTAX_STATE = state
        return new_prompt

    args = (
        self._optimizer.target['encoder']['prompt']['prompt'],
        grads['encoder']['prompt']['prompt'],
    )
    new_prompt = hcb.call(local_update, args, result_shape=args[1])

    params = self._optimizer.target
    params = params.unfreeze()
    params['encoder']['prompt']['prompt'] = new_prompt
    params = flax.core.freeze(params)

    new_state = self._optimizer.state.replace(step=self._optimizer.state.step + 1)
    new_optimizer = self._optimizer.replace(target=params, state=new_state)

    # the following should be our `OptaxOptimizer`, which does nothing.
    #  new_optimizer = self._optimizer.apply_gradient(grads, learning_rate=learning_rate)
    return self.replace(_optimizer=new_optimizer, flax_mutables=flax_mutables)

  def replace_params(self, params: VariableDict) -> 'FlaxOptimTrainState':
    return self.replace(_optimizer=self._optimizer.replace(target=params))

  def replace_flax_mutables(self,
                            flax_mutables: FrozenDict) -> 'FlaxOptimTrainState':
    return self.replace(flax_mutables=flax_mutables)

  def replace_step(self, step: jnp.ndarray) -> 'FlaxOptimTrainState':
    state_dict = self.state_dict()
    state_dict['state']['step'] = step
    return self.restore_state(state_dict)

  def restore_state(self, state_dict: VariableDict) -> 'FlaxOptimTrainState':
    new_optimizer = self._optimizer.restore_state(state_dict)
    return self.replace(
        _optimizer=new_optimizer,
        flax_mutables=flax.core.freeze(state_dict['flax_mutables'])
        if 'flax_mutables' in state_dict else EMPTY_DICT)

  def as_logical_axes(self) -> 'FlaxOptimTrainState':
    if not hasattr(self._optimizer.optimizer_def, 'derive_logical_axes'):
      raise ValueError(
          f"Optimizer '{self._optimizer.optimizer_def.__class__.__name__}' "
          'requires a `derive_logical_axes` method to be used with named axis '
          'partitioning.')
    flax_mutables_axes = self.flax_mutables_axes or EMPTY_DICT
    return FlaxOptimTrainState(
        _optimizer=self._optimizer.optimizer_def.derive_logical_axes(
            self._optimizer,
            flax_partitioning.get_axis_names(self.params_axes)),
        flax_mutables=flax_partitioning.get_axis_names(flax_mutables_axes))


class InferenceState(flax.struct.PyTreeNode):
  """State compatible with FlaxOptimTrainState without optimizer state."""

  step: jnp.ndarray
  params: flax_scope.FrozenVariableDict
  params_axes: Optional[flax_scope.FrozenVariableDict] = None
  flax_mutables: flax_scope.FrozenDict = EMPTY_DICT
  flax_mutables_axes: Optional[flax_scope.FrozenVariableDict] = None

  @classmethod
  def create(cls, model_variables: FrozenVariableDict) -> 'InferenceState':
    other_variables, params = model_variables.pop('params')
    if 'params_axes' in other_variables:
      other_variables, params_axes = other_variables.pop('params_axes')
      _validate_params_axes(params_axes, params)
    else:
      params_axes = None

    # Split other_variables into mutables and their corresponding axes.
    flax_mutables, flax_mutables_axes = _split_variables_and_axes(
        other_variables)
    flax_mutables_axes = flax_mutables_axes or None
    return InferenceState(
        step=jnp.array(0),
        params=params,
        params_axes=params_axes,
        flax_mutables=flax_mutables,
        flax_mutables_axes=flax_mutables_axes)

  @property
  def param_states(self) -> FrozenVariableDict:
    """The optimizer states of the parameters as a PyTree."""
    raise NotImplementedError('InferenceState has no optimizer states.')

  def apply_gradient(self, *args, **kwargs) -> 'InferenceState':
    raise NotImplementedError(
        'InferenceState does not support `apply_gradient`.')

  def state_dict(self) -> MutableMapping[str, Any]:
    state_dict = {
        'target': flax.core.unfreeze(self.params),
        'state': {
            'step': self.step
        }
    }
    if self.flax_mutables:
      state_dict['flax_mutables'] = flax.core.unfreeze(self.flax_mutables)
    return state_dict

  def replace_step(self, step: jnp.ndarray) -> 'InferenceState':
    return self.replace(step=step)

  def replace_params(self, params: FrozenVariableDict) -> 'InferenceState':
    return self.replace(params=params)

  def replace_flax_mutables(self,
                            flax_mutables: FrozenDict) -> 'InferenceState':
    return self.replace(flax_mutables=flax_mutables)

  def restore_state(self, state_dict: Mapping[str, Any]) -> 'InferenceState':
    return self.replace(
        params=flax.core.freeze(state_dict['target']),
        step=state_dict['state']['step'],
        flax_mutables=flax.core.freeze(state_dict['flax_mutables'])
        if 'flax_mutables' in state_dict else EMPTY_DICT)

  def as_logical_axes(self) -> 'InferenceState':
    # Set step to None so that when the logical axes are processed by the
    # flax.partitioning.logical_to_mesh_axes function, it will be skipped
    # because jax.tree_map will short circut and never call the function on the
    # step.
    flax_mutables_axes = self.flax_mutables_axes or EMPTY_DICT
    return InferenceState(
        step=None,
        params=flax_partitioning.get_axis_names(self.params_axes),
        flax_mutables=flax_partitioning.get_axis_names(flax_mutables_axes))
