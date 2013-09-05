"""Implements posterior inference.

Does Gibbs updating of model parameters.
The 'state' argument of all methods refers to a State object to be updated,
which is modified in-place.
"""

from __future__ import division

import copy

from numpy import random

import cDists
import cPosteriors
import dists
import mh


def UpdateTopics(state):
  """Resample weights related to the inferred schema.

  """
  state.topic_sub_weights = dists.SampleTopicSubWeights(state)
  state.topic_rel_weights = dists.SampleTopicRelWeights(state)
  state.type_weights = dists.SampleTypeWeights(state)
  state.type_sub_weights = dists.SampleTypeSubWeights(state)
  #  Per-table topic inference is current unused due to erratic inference results.
  # TODO(malmaud):  Get per-table topic inference back operational.
  # cPosteriors.update_topics(state)


def UpdateRows(state):
  """Updates the row indicator variables.

  Given the topic assignment of each table,
  the latent relation of each column,
  and the observed contents of each cell,
  resample the latest row entity for each row in each table
  """

  _, new_zs = cPosteriors.UpdateRowPosterior(state)
  state.rows['zs'] = new_zs


def UpdateCols(state):
  """Updates the cluster indicator variables.
  """
  cPosteriors.UpdateColumnPosterior(state)


def UpdateRels(state):
  """Updates the relation cluster ranges.

  Update the range of each relation, given the values of objects of the
  relations and the priors over column ranges
  """
  ref_posterior = cPosteriors.CalculateRangeRefPosterior(state)
  state.rels['range_ref'] = cDists.SampleCategoricalLnArray(ref_posterior)
  cPosteriors.RangeRealPosterior(state)


def UpdateFacts(state):
  """Updates the target of each (s, r) pair.

  Given the range of each relationship,
  the observed rendered objects of each fact,
  resample the object of each (subject, relation) pair
  """
  d = cPosteriors.FactPosteriorStats(state)
  stats_sum = d['sums']
  stats_len = d['lens']
  var_data = state.hypers['sd_xo'] ** 2
  f = state.GetFactsWithRanges()
  f_active = f.active == 1
  new_facts = dists.SampleNormalPosterior(f[f_active].range_real_mu_mu,
                                          f[f_active].range_real_sd**2,
                                          var_data, stats_sum[f_active],
                                          stats_len[f_active])
  state.facts.loc[f_active, 'real'] = new_facts
  ref_posterior = cPosteriors.FactRefPosterior(state)
  state.facts.ref = cDists.SampleCategoricalLnArray(ref_posterior)


def UpdateHypers(state):
  """Update the top-level hyperparameters.

  Not currently used.
  """
  # TODO(malmaud): Be more efficient with unnecessary copying
  new_state = copy.deepcopy(state)
  w = random.randn() * .5
  new_state.hypers['alpha'] = state.hypers['alpha'] + w
  if mh.ShouldAcceptState(state, new_state):
    state.hypers['alpha'] = new_state.hypers['alpha']
    state.accept_log.append(True)
  else:
    state.accept_log.append(False)


def UpdateTokens(state):
  """Resample the token weights.

  The explicit weights are not used during inference, so this is for
  debugging/visualization purposes. Instead, the weights are marginalized out
  and only the counts are used.
  """
  state.sub_token_weights = dists.SampleSubTokenWeights(state)
  state.rel_token_weights = dists.SampleRelTokenWeights(state)


def UpdateState(state):
  """Perform one full Gibbs sweep of inference.
  """
  #  For now, hypers are set manually for each dataset so the below is
  #  commented out.
  # update_hypers(state)
  UpdateRows(state)
  UpdateTokens(state)
  UpdateCols(state)
  UpdateRels(state)
  UpdateTopics(state)
  UpdateFacts(state)




