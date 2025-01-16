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

"""Mask schedule functions R."""
import math
import jax
import jax.numpy as jnp


def schedule(ratio, total_unknown, method="cosine"):
  """Generates a mask rate by scheduling mask functions R.

  Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. During
  training, the input ratio is uniformly sampled; during inference, the input
  ratio is based on the step number divided by the total iteration number: t/T.
  Based on experiments, we find that masking more in training helps.
  Args:
    ratio: The uniformly sampled ratio [0, 1) as input.
    total_unknown: The total number of tokens that can be masked out. For
      example, in MaskGIT, total_unknown = 256 for 256x256 images and 1024 for
      512x512 images.
    method: implemented functions are ["uniform", "cosine", "pow", "log", "exp"]
      "pow2.5" represents x^2.5, "cubic_alpha-12_a-2_b-0.25" represents 12x^2 * (1-x)^0.25

  Returns:
    The mask rate (float).
  """
  if method == "uniform":
    mask_ratio = 1. - ratio
  elif "pow" in method:
    exponent = float(method.replace("pow", ""))
    mask_ratio = 1. - ratio**exponent
  elif method == "cosine":
    mask_ratio = jax.lax.cos(math.pi / 2. * ratio)
  elif method == "log":
    mask_ratio = -jnp.log2(ratio) / jnp.log2(total_unknown)
  elif method == "exp":
    mask_ratio = 1 - jnp.exp2(-jnp.log2(total_unknown) * (1 - ratio))
  elif "cubic" in method:
    alpha = float(method.split("_")[1].replace("alpha-", ""))
    a = float(method.split("_")[2].replace("a-", ""))
    b = float(method.split("_")[-1].replace("b-", ""))
    mask_ratio = alpha * (1 - ratio)**a * ratio**b
  else:
    raise NotImplementedError(f"Method {method} not implemented.")
  # Clamps mask into [epsilon, 1)
  mask_ratio = jnp.clip(mask_ratio, 1e-6, 1.)
  return mask_ratio
