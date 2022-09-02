
import adafactor
import jax.numpy as jnp


class OptaxOptimizer(adafactor.Adafactor):

    def __init__(self, *args, **kwargs):
        super(OptaxOptimizer, self).__init__(*args, **kwargs)

    def apply_param_gradient(self, step, hyper_params, param, state, grad, path):
        assert hyper_params.learning_rate is not None, 'no learning rate provided.'

        # unpack parameters
        learning_rate = hyper_params.learning_rate
        beta1 = hyper_params.beta1
        decay_rate = hyper_params.decay_rate
        step_offset = hyper_params.step_offset
        multiply_by_parameter_scale = hyper_params.multiply_by_parameter_scale
        max_parameter_scale = hyper_params.max_parameter_scale
        clipping_threshold = hyper_params.clipping_threshold
        weight_decay_rate = hyper_params.weight_decay_rate
        epsilon1 = hyper_params.epsilon1
        epsilon2 = hyper_params.epsilon2
        if hyper_params.weight_decay_rate_lr_exponent:
          weight_decay_rate = (
              (weight_decay_rate or 1.0) *
              learning_rate**hyper_params.weight_decay_rate_lr_exponent)

        if self.hyper_params.factored:
          factor_rule = (
              self.hyper_params.factor_map[path]
              if self.hyper_params.factor_map else adafactor.HEURISTIC_RULE)
        else:
          factor_rule = None
        averaging_dims, factored_dims = self._parse_rule(factor_rule, param.shape,
                                                         path)

        # compute update
        grad = grad.astype(jnp.float32)
        updates = {k: jnp.zeros((1,)) for k in ['v_row', 'v_col', 'v', 'm']}
        decay_rate = self._decay_rate_pow(step - step_offset, exponent=decay_rate)
        update_scale = learning_rate

        if isinstance(multiply_by_parameter_scale, adafactor.HParamMap):
          multiply_by_parameter_scale = multiply_by_parameter_scale[path]
        if multiply_by_parameter_scale:
          param_scale = jnp.sqrt(
              jnp.mean(param * param, axis=averaging_dims, keepdims=True))
          # Clip param_scale to a minimum value of epsilon2.
          param_scale = jnp.maximum(param_scale, epsilon2)
          # Clip param_scale to a maximum value, if specified.
          if max_parameter_scale is not None:
            param_scale = jnp.minimum(param_scale, max_parameter_scale)
          update_scale *= param_scale
        mixing_rate = 1.0 - decay_rate

        grad_sqr = grad * grad + epsilon1
        if factored_dims is adafactor.HEURISTIC_RULE:
          factored_dims = self._factored_dims(param.shape)
        if factored_dims is not None:
          d1, d0 = factored_dims
          new_v_row = (
              decay_rate * state.v_row + mixing_rate * jnp.mean(grad_sqr, axis=d0))
          new_v_col = (
              decay_rate * state.v_col + mixing_rate * jnp.mean(grad_sqr, axis=d1))
          updates['v_row'] = new_v_row
          updates['v_col'] = new_v_col
          reduced_d1 = tuple(d - len([e for e in d0 if e < d]) for d in d1)

          row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
          row_factor = (new_v_row / row_col_mean)**-0.5
          col_factor = (new_v_col)**-0.5
          y = (
              grad * jnp.expand_dims(row_factor, axis=d0) *
              jnp.expand_dims(col_factor, axis=d1))
        else:
          new_v = decay_rate * state.v + mixing_rate * grad_sqr
          updates['v'] = new_v
          y = grad * (new_v)**-0.5

        if clipping_threshold is not None:
          clipping_denom = (
              jnp.maximum(
                  1.0,
                  jnp.sqrt(jnp.mean(y * y, axis=averaging_dims, keepdims=True)) /
                  clipping_threshold))
          y /= clipping_denom

        subtrahend = update_scale * y
        if beta1 is not None:
          new_m = beta1 * state.m + (1.0 - beta1) * subtrahend
          subtrahend = new_m
          updates['m'] = new_m.astype(self.dtype_momentum)

        if weight_decay_rate is not None:
          new_param = (1.0 - weight_decay_rate) * param - subtrahend
        else:
          new_param = param - subtrahend

        if hyper_params.skip_nan_updates:
          updates['v_row'] = jnp.where(
              jnp.isnan(updates['v_row']), state.v_row, updates['v_row'])
          updates['v_col'] = jnp.where(
              jnp.isnan(updates['v_col']), state.v_col, updates['v_col'])
          updates['v'] = jnp.where(jnp.isnan(updates['v']), state.v, updates['v'])
          updates['m'] = jnp.where(jnp.isnan(updates['m']), state.m, updates['m'])
          new_param = jnp.where(jnp.isnan(new_param), param, new_param)
        new_state = adafactor._AdafactorParamState(**updates)

        return new_param.astype(param.dtype), new_state

