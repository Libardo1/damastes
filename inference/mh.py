"""Implementation of Metropolis-Hastings moves.
"""

from __future__ import division
import numpy as np
from numpy import random

import dists


def ShouldAccept(loglik_before, loglik_after, hastings=0.):
  """Using the Metropolis-Hastings rule to decide whether to accept a proposal.

  Args:
    llh_before: Log-likelihood of the model before applying a proposal.
    llh_after: Log-likelihood of the model after
    proposing a change to its latent state.

  Returns:
    True if the proposal should be accepted

  See http://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm.
  """
  delta = loglik_after - loglik_before + hastings
  r = np.log(random.rand())
  return delta > r


def ShouldAcceptState(state_before, state_after):
  """Decides whether to accept a proposed state.

  Most generic MH transition function for symmetric proposals to
  any set of model variables.
  Assumes symmetric proposal (hence Hastings correction factor is unnecessary)
  Currently calculates the complete log-likelihood
  of the model; no smart caching.

  Args:
    state_before: The current state.
    state_after: The proposed new state.

  Returns:
    A boolean indicating whether to accept the new state.
  """
  llh_before = dists.CalculateLogLikelihoodOfAllData(state_before)
  llh_after = dists.CalculateLogLikelihoodOfAllData(state_after)
  if llh_after == -np.inf:
    return False
  return ShouldAccept(llh_before, llh_after)

