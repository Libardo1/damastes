"""Methods for sampling model parameters from their prior distributions.

Useful for Geweke testing and debugging.

Can be considered the 'reference implementation' of the model to which
the posterior inference correctness can be compared.

See
https://docs.google.com/a/google.com/document/d/1MJTTWiMSWzVuLph7-6usaYD0otW-GRIaqT5QBYwuAak/edit
for description of model and its constituent latent variables.

Currently only the distribution over relations is handled in an explicitly
non-parametric fashion. The distribution over subjects and tokens is approximated
as non-parametric through a crude finite-truncation approximation.
"""


from __future__ import division

import logging

import numpy as np
from numpy import random
import pandas

import cCollect
import cDists
import state
import util


def SamplePrior(data_dims):
  """Draw a sample from the prior.

  Args:
    desc: The dimensions of the observed data to simulate. desc should have one row per table to simulate. Each row should be a tuple of the form (n_rows, n_cols) that specifies the dimensionality of the corresponding table.

  Returns:
    A State object
  """
  prior_sample = state.State()
  prior_sample.K = state.SIZE
  prior_sample.SetSize(data_dims)
  UpdateHypersFromPrior(prior_sample)
  UpdateModelParamsFromPrior(prior_sample)
  UpdateRelationsFromPrior(prior_sample)
  # The below calls to 'reset_index' cause the the indices to also appear
  # as regular columns in the dataframes, which the rest of the module expects.
  prior_sample.facts.reset_index(inplace=True, drop=False)
  prior_sample.rows.reset_index(inplace=True, drop=False)
  prior_sample.cols.reset_index(inplace=True, drop=False)
  UpdateFactsFromPrior(prior_sample)
  UpdateTopicsFromPrior(prior_sample)
  UpdateRowsFromPrior(prior_sample)
  UpdateColsFromPrior(prior_sample)
  UpdateTypesFromPrior(prior_sample)
  prior_sample.ConvertToThirdNormalForm()
  UpdateObservationsFromPrior(prior_sample)
  return prior_sample


def ResampleLatents(state):
  UpdateHypersFromPrior(state)
  UpdateModelParamsFromPrior(state)
  UpdateRelationsFromPrior(state)
  #UpdateFactsFromPrior(state)
  UpdateTopicsFromPrior(state)
  UpdateRowsFromPrior(state)
  UpdateColsFromPrior(state)


# These Update functions all take a State object and update it in-place

def UpdateModelParamsFromPrior(state):
  alpha = state.hypers['alpha']
  alpha_token = state.hypers['alpha_token']
  state.topic_sub_weights = (
      cDists.SampleDirichletArray(alpha * np.ones((state.n_topics, state.n_subs))))
  state.topic_rel_weights = (
      cDists.SampleDirichletArray(alpha * np.ones((state.n_topics, state.n_rels))))
  state.type_sub_weights = (
      cDists.SampleDirichletArray(alpha * np.ones((state.n_types, state.n_subs))))
  try:
    state.sub_token_weights = (
        cDists.SampleDirichletArray(alpha_token * np.ones((state.n_subs, state.n_tokens))))
    state.rel_token_weights = (
        cDists.SampleDirichletArray(alpha_token * np.ones((state.n_rels, state.n_tokens))))
  except ZeroDivisionError:
    # If alpha is extremely small, approximate the Dirichlet draw
    # with a delta function
    logging.warning('Token alpha %r very small. Approximating with Dirichlet draw.',
                    alpha_token)
    eps = 1e-4  # Small constant to prevent numeric instability
    sub_assign = random.randint(state.n_tokens, size=state.n_subs)
    ent_token_weights = np.zeros((state.n_subs, state.n_tokens)) + eps
    ent_token_weights[np.arange(state.n_subs), sub_assign] = 1. - eps
    state.sub_token_weights = ent_token_weights
    rel_assign = random.randint(state.n_rels, size=state.n_rels)
    rel_token_weights = np.zeros((state.n_rels, state.n_tokens)) + eps
    rel_token_weights[np.arange(state.n_rels), rel_assign] = 1. - eps
    state.rel_token_weights = rel_token_weights
  state.type_names = pandas.Series({0: 'ref', 1: 'real'})
  state.topic_weights = random.dirichlet(np.repeat(alpha, state.n_topics))
  state.type_weights = random.dirichlet(np.repeat(alpha, state.n_types))


def UpdateHypersFromPrior(state):
  rel_range = cDists.normal_gamma_parms(mu=10., nu=.5, shape=1., rate=10.)
  hypers = dict(sd_xo=.1, alpha_shape=10, alpha_scale=1, rel_range=rel_range)
  # dp_alpha not currently used since alpha hyperparameters are being manually set
  dp_alpha = random.gamma(hypers['alpha_shape'], hypers['alpha_scale'])
  hypers['alpha'] = 1.
  hypers['alpha_sub'] = 1.
  hypers['alpha_token'] = .001
  hypers['alpha_zr'] = 1.
  state.hypers = pandas.Series(hypers)


def UpdateFactsFromPrior(state):
  facts = state.GetFactsWithRanges()
  mu = facts.range_real_mu
  sd = facts.range_real_sd
  new_real = random.normal(mu, sd)
  state.facts['real'] = new_real
  w = np.log(state.type_sub_weights[facts.range_ref, :])
  state.facts['ref'] = cDists.SampleCategoricalLnArray(w)


def UpdateRelationsFromPrior(state):
  r = state.rels
  rel_range = state.hypers['rel_range']
  # random.gamma uses the 'shape', 'scale' paramterization of the gamma distribution.
  # rel_range is expressed in terms of the 'shape', rate' parameterization.
  # To convert, we need to invert the rate to obtain the scale.
  # See http://en.wikipedia.org/wiki/Gamma_distribution
  prec = random.gamma(rel_range.shape, scale=1./rel_range.rate, size=len(r))
  mu_sd = util.TauToSD(prec * rel_range.nu)
  # Each relation has a corresponding distribution over both numeric-valued
  # and entity-valued data. The latent relation of a column and the type
  # of the column jointly determine the distribution over the cells in that column.
  r.loc[:, 'range_real_sd'] = util.TauToSD(prec)
  r.loc[:, 'range_real_mu'] = random.normal(rel_range.mu, mu_sd)
  r.loc[:, 'range_ref'] = cDists.SampleCategoricalArray(state.type_weights, len(r))


def UpdateRowsFromPrior(state):
  w = state.topic_sub_weights
  row_topics = state.tables.zt.values[state.rows.table.values]
  for topic in range(state.n_topics):
    rows_for_topic = row_topics == topic
    n = np.sum(rows_for_topic)
    state.rows.loc[rows_for_topic, 'zs'] = cDists.SampleCategoricalArray(w[topic, :], n)


def UpdateColsFromPrior(state):
  col_topics = state.tables.zt.values[state.cols.table.values]
  alpha = state.hypers['alpha_zr']
  for topic in range(state.n_topics):
    cols_for_topic = col_topics == topic
    n = sum(cols_for_topic)
    if n==0:
      continue
    state.cols.loc[cols_for_topic, 'zr'] = cDists.SampleCRPPrior(n, alpha)
  # We compute how many columns belong to each latent relation.
  # Relations which have no members are marked as non-active.
  sizes = state.cols.groupby('zr').size()
  is_active = sizes.index
  state.rels.loc[:, 'active'] = 0
  state.rels.loc[is_active, 'active'] = 1


def UpdateTypesFromPrior(state, force_type=None):
  if force_type:
    state.cols['type'] = force_type
  else:
    # Each column can be one of two types.
    # '0' indicates the column contains references to other entities.
    # '1' indicates the column contains numeric data interpreted as literal real values.
    state.cols['type'] = random.randint(0, 2, size=len(state.cols))


def UpdateTopicsFromPrior(state):
  weights = state.topic_weights
  topics = cDists.SampleCategoricalArray(weights, len(state.tables))
  state.tables.zt = topics


def UpdateObservationsFromPrior(state):
  facts = cCollect.JoinCellsWithFactsTable(state)
  real_type = facts['type'] == 1
  ref_type = facts['type'] == 0
  state.cells.loc[real_type, 'xo_real'] = (
      random.normal(facts['real'], state.hypers['sd_xo'])[real_type])
  token_weights = np.log(state.sub_token_weights[facts['ref'], :])
  state.cells.loc[ref_type, 'xo_ref'] = (
      cDists.SampleCategoricalLnArray(token_weights)[ref_type])
  token_weights = np.log(state.sub_token_weights[state.rows.zs])
  state.rows.xs = cDists.SampleCategoricalLnArray(token_weights)
  token_weights = np.log(state.rel_token_weights[state.cols.zr])
  state.cols.xr = cDists.SampleCategoricalLnArray(token_weights)


def UpdateObservationsAndTypesFromPrior(state):
  UpdateTypesFromPrior(state)
  UpdateObservationsFromPrior(state)
