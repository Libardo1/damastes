"""Methods for sampling from the distributions involved in this model.

Generally these work by taking a state object and manipulating part
 of it in-place.
Function naming method inspired by pymc
"""
from __future__ import division

import cCollect
import cDists
import numpy as np
from numpy import random
import pandas
from scipy import stats


def SampleTopicSubWeights(state):
  """Update the distribution over subjects for each table topic.

  Computes P(zs|zt) for each zs, zt.
  The return value is a matrix such that weights[i,j] = P(zs=j|zt=i)
  """
  alpha = state.hypers['alpha_sub']
  counts = cCollect.TabulateTopicSubjectCounts(state) + alpha
  weights = cDists.SampleDirichletArray(counts)
  return weights


def SampleSubTokenWeights(state):
  alpha = state.hypers['alpha_token']
  counts = cCollect.TabulateSubjectTokenCounts(state) + alpha
  weights = cDists.SampleDirichletArray(counts)
  return weights


def SampleRelTokenWeights(state):
  alpha = state.hypers['alpha_token']
  counts = cCollect.TabulateRelationTokenCounts(state) + alpha
  weights = cDists.SampleDirichletArray(counts)
  return weights


def SampleTopicRelWeights(state):
  """Computes P(zr|zt) for each zr, zt.
  The return value is a matrix such that weights[i,j] = P(zr=j|zt=i)
  """
  alpha = state.hypers['alpha']
  counts = cCollect.TabulateTopicRelationCounts(state) + alpha
  weights = cDists.SampleDirichletArray(counts)
  return weights


def SampleTopicWeights(state):
  obs_counts = np.bincount(state.rels.range_ref, minlength=state.n_topics)
  alpha = state.hypers['alpha']
  counts = obs_counts + alpha
  weights = random.dirichlet(counts)
  return weights


def SampleTypeWeights(state):
  obs_counts = cCollect.TabulateRangeCounts(state)
  alpha = state.hypers['alpha']
  weights = random.dirichlet(obs_counts + alpha)
  return weights


def SampleTypeSubWeights(state):
  obs_counts = cCollect.TabulateRangeSubjectCounts(state)
  alpha = state.hypers['alpha']
  weights = cDists.SampleDirichletArray(obs_counts + alpha)
  return weights


def CalculateNormalLogLikelihood(x, mu, sd):
  """Calculates unnormalized log-likelihoods."""
  x_c = x - mu
  return x_c * x_c * -.5 * 1/(sd ** 2)


def CalculateLogLikelihoodOfAllData(state, return_by_parts=False):
  """Returns the joint log-likelihood of a state.

  Used by Metropolis-Hastings to accept or reject proposed changes to a state.

  Also useful as a diagnostic for roughly assessing mixing time when computed
  as a marginal statistic over a trace.

  Args:
    state: A State object containing latent model parameters and observed data
    return_by_parts: If True, return a tuple of likelihoods,
      with one entry per latent variable of the model. Useful for debugging.
      Otherwise, return a single scalar that is the joint likelihood
      of the entire model.
  """
  # TODO(malmaud): Add comments explaining the math in each calculation
  # TODO(malmaud): Update to be accurate for recent changes to the model
  hypers = state.hypers
  alpha = hypers['alpha']
  dp_alpha = alpha * state.K
  alpha_llh = stats.gamma.logpdf(dp_alpha, hypers['alpha_shape'],
                                 scale=hypers['alpha_scale'])
  topic_counts = np.bincount(state.tables.zt, minlength=state.n_topics)
  topic_llh = (
      cDists.CalculateDirichletMultinomialLikelihood(topic_counts,
                                                     np.repeat(alpha, state.n_topics)))
  mu = state.rels.loc[state.facts.zr].range_real_mu_mu
  facts_llh = np.sum(CalculateNormalLogLikelihood(state.facts.real.values-mu.values,
                                                  0.,
                                                  state.hypers['sd_fact']))
  rels_llh = np.sum(CalculateNormalLogLikelihood(state.rels.range_real_mu_mu,
                                                 0.,
                                                 state.hypers['sd_rel_range']))
  topic_sub_llh = np.empty(state.n_topics)
  r = state.rows.join(state.tables, on='table').set_index('zt').sort_index()
  c = state.cols.join(state.tables, on='table').set_index('zt').sort_index()
  for zt in range(state.n_topics):
    if zt not in r.index:  # topic not in use.
      topic_sub_llh[zt] = 0.
      continue
    zs = r.loc[zt, 'zs']
    if np.isscalar(zs):
      zs = [zs]
    zs_counts = np.bincount(zs, minlength=state.n_subs)
    zs_llh = (
        cDists.CalculateDirichletMultinomialLikelihood(zs_counts,
                                                       np.repeat(alpha, state.n_subs)))
    zr = c.loc[zt, 'zr']
    if np.isscalar(zr):
      zr = [zr]
    zr_counts = np.bincount(zr, minlength=state.n_rels)
    zr_llh = (
        cDists.CalculateDirichletMultinomialLikelihood(zr_counts,
                                                       np.repeat(alpha, state.n_rels)))
    topic_sub_llh[zt] = zs_llh + zr_llh
  cells = state.GetCellsJoinedWithFactsTable()
  xo_llh = np.sum(CalculateNormalLogLikelihood(cells.real-cells.xo_real, 0.,
                                               state.hypers['sd_xo']))
  llh_set = pandas.Series([alpha_llh, topic_llh, facts_llh, rels_llh,
                           np.sum(topic_sub_llh), xo_llh],
                          index=['alpha', 'topic', 'facts',
                                 'rels', 'topic_sub', 'xo'])
  if return_by_parts:
    return llh_set
  else:
    return np.sum(llh_set)


def NormalPosteriorStats(mu_prior, var_prior, var_data, sum_data, n_data):
  """Returns a new mean and variance from a normal prior with known variance.

  Returns:
    Pandas dataseries with updated means and variances

  See the table at http://en.wikipedia.org/wiki/Conjugate_prior.
  This case corresponds to 'Normal with known variance'.
  """
  normalizer = 1./var_prior + n_data/var_data
  mu = mu_prior/var_prior + sum_data/var_data
  mu /= normalizer
  var = 1./normalizer
  return pandas.Series([mu, var], index=['mu', 'var'])


def SampleNormalPosterior(mu0, var0, var_data, sum_data, n_data):
  data_stats = NormalPosteriorStats(mu0, var0, var_data, sum_data, n_data)
  return random.normal(data_stats['mu'], np.sqrt(data_stats['var']))
