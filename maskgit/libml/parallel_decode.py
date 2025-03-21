# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fast decoding routines for non-autoregressive generation."""

import flax
import jax
import jax.numpy as jnp
from jax import lax
from jax_tqdm import loop_tqdm

from maskgit.libml import mask_schedule

# Confidence score for known tokens to avoid masking or re-predicting them.
# Here we don't use 1.0 because the upper bounder of the probability can be
# possibility larger than 1 due to the noise addition.
_CONFIDENCE_OF_KNOWN_TOKENS = jnp.inf


def mask_by_random_topk(rng, mask_len, probs, temperature=1.0):
  """Modifies from jax.random.choice without replacement.

  JAX's original implementation is as below:
    g = -gumbel(key, (n_inputs,)) - jnp.log(p)
    ind = jnp.argsort(g)[:n_draws]
  We add temperature annealing on top of it, which is:
    g = -gumbel(key, (n_inputs,)) - temperature * jnp.log(p)
    ind = jnp.argsort(g)[:n_draws]

  Args:
    rng: a PRNG key used as the random key.
    mask_len: the number to mask.
    probs: the probabilities associated with each entry.
    temperature: when temperature = 1.0, it's identical to jax's implementation.
      The larger this value is, the more random the masking is picked.

  Returns:
    A binary masking map [batch_size, seq_len].
  """
  confidence = jnp.log(probs) + temperature * jax.random.gumbel(rng, probs.shape)
  sorted_confidence = jnp.sort(confidence, axis=-1)
  # Obtains cut off threshold given the mask lengths.
  cut_off = jnp.take_along_axis(sorted_confidence, mask_len.astype(jnp.int32), axis=-1)
  # Masks tokens with lower confidence.
  masking = (confidence < cut_off)
  return masking


@flax.struct.dataclass
class State:
  """Holds decoding state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: jnp.ndarray  # scalar int32: current decoded length index
  # The active sequence log probabilities and finished sequence scores.
  cur_seqs: jnp.ndarray  # int32 [batch, seq_len]
  # The logprob of the decoded token (at time of decoding); set to -inf for masked tokens
  cur_neg_logprobs: jnp.ndarray  # float32 [batch, seq_len]
  rng: jnp.ndarray  # Sampling random state.
  final_seqs: jnp.ndarray  # int32 [batch, num_iter, seq_len]


def state_init(init_indices, rng, num_iter, start_iter=0):
  """Initializes the decoding state data structure."""
  cur_index0 = jnp.array(start_iter)
  cur_seqs0 = init_indices
  cur_neg_logprobs0 = jnp.ones_like(init_indices, dtype=jnp.float32) * -_CONFIDENCE_OF_KNOWN_TOKENS
  final_seqs0 = jnp.expand_dims(init_indices, 1)
  final_seqs0 = jnp.tile(final_seqs0, (1, num_iter, 1))
  return State(
    cur_index=cur_index0,
    cur_seqs=cur_seqs0,
    cur_neg_logprobs=cur_neg_logprobs0,
    rng=rng,
    final_seqs=final_seqs0)

def decode(inputs,
           rng,
           tokens_to_logits,
           decoding_strategy='maskgit',
           mask_token_id=-1,
           num_iter=12,
           start_iter=0,
           choice_temperature=1.0,
           sampling_temperature=1.0,
           sampling_temperature_annealing=False,
           mask_scheduling_method="cosine",
           remdm_eta=0.0):
  """Fast decoding for iterative generation.

  Args:
    inputs: int32 array: [batch_size, seq_length] input sequence of masked
      tokens, where the masked tokens are defined by mask_token_id.
    rng: jnp.ndarray: sampling random state.
    tokens_to_logits: decoder function taking single token slices and cache and
      returning logits and updated cache.
    decoding_strategy: string: decoding strategy to use.
      Options are 'maskgit', 'remdm', 'mdlm'.
    mask_token_id: int: [Mask] token id.
    num_iter: int: default is 12.
    start_iter: int: default is 0.
    choice_temperature: float: temperature to control the randomness of masking.
    sampling_temperature: float: temperature to control the randomness of token id sampling.
    sampling_temperature_annealing: bool: whether to use sampling temperature annealing.
    mask_scheduling_method: masking method string. See mask_schedule.py for
      details.
    remdm_eta: float: eta value for REMDM constant / loop decoding strategy.

  Returns:
     [batch_size, num_iter, seq_length] output sequence of tokens in all
       iterations.
  """
  inputs = inputs.astype("int32")
  unknown_number_in_the_beginning = jnp.sum(inputs == mask_token_id, axis=-1)
  # Initializes state
  init_state = state_init(inputs, rng, num_iter, start_iter=start_iter)

  def loop_cond_fn(state):
    """Beam search loop termination condition."""
    not_at_end = (state.cur_index < num_iter)
    return not_at_end

  @loop_tqdm(num_iter, leave=False)
  def loop_body_fn(step: int, state: State) -> State:
    """Beam search loop state update function."""
    loop_rng = state.rng
    # step = state.cur_index
    # Current input ids: [batch_size, seq_length].
    cur_ids = state.cur_seqs

    # Calls model on current seqs to get next-iteration seqs.
    if sampling_temperature_annealing:
      temp = sampling_temperature * (num_iter - step) / num_iter
    else:
      temp = sampling_temperature
    logits = tokens_to_logits(cur_ids) / temp
    vocab_size = logits.shape[-1]

    ####### REMDM ########
    if 'remdm' in decoding_strategy:
      # Compute alpha
      t = 1. * (num_iter - step) / num_iter
      alpha_t = mask_schedule.schedule(t, unknown_number_in_the_beginning, mask_scheduling_method)
      s = 1. * (num_iter - step - 1) / num_iter
      alpha_s = mask_schedule.schedule(s, unknown_number_in_the_beginning, mask_scheduling_method)

      # Forward pass
      logits = logits.at[..., mask_token_id].set(-_CONFIDENCE_OF_KNOWN_TOKENS)
      x_theta = jax.nn.softmax(logits, axis=-1)

      if decoding_strategy == 'remdm_conf':
        # Compute sigma per token: sigma = 0 --> MDLM; sigma = max_sigma --> move as much mass away from z_t as possible
        max_sigma = jnp.minimum((1 - alpha_s) / alpha_t, 1.)
        eta = jax.nn.softmax(state.cur_neg_logprobs, axis=-1)
        eta = jnp.nan_to_num(jnp.where((cur_ids == mask_token_id), 0., eta))
        sigma = eta[..., None] * max_sigma
      elif decoding_strategy == 'remdm_rescale':
        # Compute sigma per token: sigma = 0 --> MDLM; sigma = max_sigma --> move as much mass away from z_t as possible
        max_sigma = jnp.minimum((1 - alpha_s) / alpha_t, 1.)
        sigma = remdm_eta * max_sigma
      elif decoding_strategy == 'remdm_cap':
        sigma = jnp.minimum((1 - alpha_s) / alpha_t, remdm_eta)
      # TODO: Implement REMDM-Loop
      else:
        raise NotImplementedError(f"REMDM decoding strategy {decoding_strategy} not implemented.")

      # Compute REMDM posterior
      limiting_distribution = jax.nn.one_hot(jnp.array([vocab_size - 1]), vocab_size)
      # Case 1: cur_ids = mask
      case1 = ((alpha_s - alpha_t * (1 - sigma)) / (1 - alpha_t) * x_theta +
               (1 - alpha_s - alpha_t * sigma) / (1 - alpha_t) * limiting_distribution)
      # Case 2: cur_ids != mask
      case2 = (1 - sigma) * jax.nn.one_hot(cur_ids, vocab_size) + sigma * limiting_distribution
      q_xs = jnp.where((cur_ids == mask_token_id)[..., None], case1, case2)

      # Sample the ids using categorical sampling: [batch_size, seq_length].
      loop_rng, sample_rng = jax.random.split(loop_rng, 2)
      gumbel_norm = 1e-10 - jnp.log(jax.random.uniform(sample_rng, q_xs.shape) + 1e-10)
      sampled_ids = (q_xs / gumbel_norm).argmax(axis=-1)
      # Replace token |V| with mask_token_id
      sampled_ids = jnp.where(sampled_ids == (vocab_size - 1), mask_token_id, sampled_ids)
      # Update decoded neg_logprobs
      neg_logprobs = -jax.nn.log_softmax(logits, axis=-1)
      selected_neg_logprobs = jnp.squeeze(
        jnp.take_along_axis(neg_logprobs, jnp.expand_dims(sampled_ids.astype(jnp.int32), -1), -1), -1)
      # Keep masked tokens' neg_logprobs as -inf
      cur_neg_logprobs = jnp.where((sampled_ids == mask_token_id), state.cur_neg_logprobs, selected_neg_logprobs)
      # (re-)Hardcode BOS token
      sampled_ids = sampled_ids.at[..., 0].set(cur_ids[..., 0])
      cur_neg_logprobs = cur_neg_logprobs.at[..., 0].set(-_CONFIDENCE_OF_KNOWN_TOKENS)  # BOS token should remain unmasked
      final_seqs = jax.lax.dynamic_update_slice(
        state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))

    ####### MDLM ########
    elif decoding_strategy == 'mdlm':
      logits = logits.at[..., mask_token_id].set(-_CONFIDENCE_OF_KNOWN_TOKENS)
      x_theta = jax.nn.softmax(logits, axis=-1)
      t = 1. * (num_iter - step) / num_iter
      move_chance_t = 1. - mask_schedule.schedule(t, unknown_number_in_the_beginning,
                                          mask_scheduling_method)
      s = 1. * (num_iter - step - 1) / num_iter
      move_chance_s = 1. - mask_schedule.schedule(s, unknown_number_in_the_beginning,
                                             mask_scheduling_method)
      q_xs = x_theta * (move_chance_t - move_chance_s) / move_chance_t
      q_xs = q_xs.at[..., mask_token_id].set(move_chance_s / move_chance_t)

      loop_rng, sample_rng = jax.random.split(loop_rng, 2)
      # Sample the ids using categorical sampling: [batch_size, seq_length].
      # sampled_ids = jax.random.categorical(sample_rng, q_xs)
      gumbel_norm = 1e-10 - jnp.log(jax.random.uniform(sample_rng, q_xs.shape) + 1e-10)
      sampled_ids = (q_xs / gumbel_norm).argmax(axis=-1)
      # Replace token |V| with mask_token_id
      sampled_ids = jnp.where(sampled_ids == (vocab_size - 1), mask_token_id, sampled_ids)
      unknown_map = (cur_ids == mask_token_id)
      sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
      final_seqs = jax.lax.dynamic_update_slice(
        state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
      cur_neg_logprobs = jnp.zeros_like(cur_ids, dtype=state.cur_neg_logprobs.dtype)

    ######## MaskGiT ########
    elif decoding_strategy == 'maskgit':
      loop_rng, sample_rng = jax.random.split(loop_rng, 2)
      # Sample the ids using categorical sampling: [batch_size, seq_length].
      sampled_ids = jax.random.categorical(sample_rng, logits)

      # Just updates the masked tokens.
      unknown_map = (cur_ids == mask_token_id)
      sampled_ids = jnp.where(unknown_map, sampled_ids, cur_ids)
      # Defines the mask ratio for the next round. The number to mask out is
      # determined by mask_ratio * unknown_number_in_the_beginning.
      ratio = 1. * (step + 1) / num_iter
      mask_ratio = mask_schedule.schedule(ratio, unknown_number_in_the_beginning,
                                          mask_scheduling_method)
      # Updates final seqs with the current sampled_ids.
      final_seqs = jax.lax.dynamic_update_slice(
          state.final_seqs, jnp.expand_dims(sampled_ids, axis=1), (0, step, 0))
      # Computes the probabilities of each selected token.
      probs = jax.nn.softmax(logits, axis=-1)
      selected_probs = jnp.squeeze(
          jnp.take_along_axis(probs, jnp.expand_dims(sampled_ids.astype(jnp.int32), -1), -1), -1)
      # Ignores the tokens given in the input by overwriting their confidence.
      selected_probs = jnp.where(unknown_map, selected_probs,
                                 _CONFIDENCE_OF_KNOWN_TOKENS)
      # Gets mask lens for each sample in the batch according to the mask ratio.
      mask_len = jnp.expand_dims(
          jnp.floor(unknown_number_in_the_beginning * mask_ratio), 1)
      # Keeps at least one of prediction in this round and also masks out at least
      # one and for the next iteration
      mask_len = jnp.maximum(
          1,
          jnp.minimum(jnp.sum(unknown_map, axis=-1, keepdims=True) - 1, mask_len))

      # Adds noise for randomness
      loop_rng, choice_rng = jax.random.split(loop_rng)
      masking = mask_by_random_topk(choice_rng, mask_len, selected_probs,
                                    choice_temperature * (1. - ratio))
      # Masks tokens with lower confidence.
      sampled_ids = jnp.where(masking, mask_token_id, sampled_ids)
      cur_neg_logprobs = jnp.zeros_like(cur_ids, dtype=state.cur_neg_logprobs.dtype)
    else:
      raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")

    return State(
        cur_index=state.cur_index + 1,
        cur_seqs=sampled_ids,
        cur_neg_logprobs=cur_neg_logprobs,
        rng=loop_rng,
        final_seqs=final_seqs)

  # Run while loop and get final beam search state.
  # final_state = lax.while_loop(loop_cond_fn, loop_body_fn, init_state)
  final_state = lax.fori_loop(0, num_iter, loop_body_fn, init_state)
  return final_state.final_seqs

