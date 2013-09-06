"""Functions for performing various ad-hoc tests.

Convenient to run from the Python command line when testing interactively,
since it imports all of the modules you'd want to interact with.
"""

from __future__ import division
import logging
import util

# The log config has to be set before importing the other modules, since
# the external modules might configure basic logging differently and it can
# only be changed once per process.
util.ConfigureLogging()



# Many of these imports are not actually used within the module, but are
# imported for the convenience of the interactive user.

from matplotlib.pylab import *
import cDists
import cCollect
import cPosteriors
from numpy import random
import pandas
import os
from pandas import Series, DataFrame
import dists
import ipdb
import geweke
from IPython.parallel import Client
from IPython.config import Application
import prior_sampler
import posterior_sampler

import copy
import mcmc
import state
import data

seterr(all='warn')

desc=[(6,2)]*3
state=prior_sampler.SamplePrior(desc)

def unique_topics(s):
  """
  How many topics are used to explain the observed data
  """
  return len(pandas.unique(s.tables.zt))


def unique_tokens(s):
  c=s.GetCellsJoinedWithFactsTable()
  return len(unique(c[c.type==0].xo_ref))


def unique_subs(s):
  return len(unique(s.rows.zs))


def unique_rels(s):
  return len(unique(s.cols.zr))


def fact_repeat(s):
  counts=s.facts.groupby('ref').size()
  return mean(counts)


def unique_refs(s):
  return len(unique(s.rels.range_ref))


def cell_means(s):
  c=s.GetCellsJoinedWithFactsTable()
  return mean(c.loc[c.type==1, 'xo_real'])


def fact_means(s):
  return mean(s.facts.real)


def test_normal_posterior():
  N=20
  s=[]
  s2=[]
  mu=0.
  sd0=5.
  sd1=2.
  for i in range(30000):
    x=random.normal(mu, sd1, size=N)
    mu=dists.SampleNormalPosterior(0, sd0**2, sd1**2, sum(x), len(x))
    mu_prior=random.normal(0,sd0)
    s.append(mu)
    s2.append(mu_prior)
  hist(s)
  print mean(s), std(s)
  print mean(s2), std(s2)


def test_normal_gamma_posterior():
  prior = cDists.normal_gamma_parms(6., .3, 6.3, .04)
  post = cDists.normal_parms()
  post.mu=0.
  post.sd=1.
  N=20
  s=[]
  s2=[]
  for i in range(30000):
    x=random.normal(post.mu, post.sd, size=N)
    post=cDists.SampleNormalGamma(cDists.NormalGammaPosteriorParms(prior, x))
    s.append([post.mu, post.sd])
    post2=cDists.SampleNormalGamma(prior)
    s2.append([post2.mu, post2.sd])
  s=array(s)
  s2=array(s2)
  print std(s,0)
  print std(s2,0)
  print mean(s,0)
  print mean(s2,0)
  return s,s2


def update_state(s):
  posterior_sampler.UpdateState(s)


def update_data(s):
  prior_sampler.UpdateObservationsAndTypesFromPrior(s)


def geweke_test(n_posterior, g=None, remote=False):
  if g is None:
    g=geweke.Geweke()
    g.Test(100, 0, desc, update_state, update_data)
  if not remote:
    g.Test(100, n_posterior, desc, update_state, update_data, skip=10)
    return g
  res = dview.apply(client_geweke, n_posterior)
  while not res.ready():
    res.wait(5)
    logging.info('Waiting for clients...')
  res.display_outputs()
  for samples in res:
    g._geweke_samples.extend(samples)
  return g


def client_geweke(n_posterior):
  import dists
  import state
  reload(state)
  reload(dists)
  import geweke
  g = geweke.Geweke()
  g.Test(0, n_posterior, desc)
  return g._geweke_samples


def client_setup():
  import sys
  if wd not in sys.path:
    sys.path.insert(0, wd)


def test_row_posterior(N=1000):
  s = prior_sampler.SamplePrior(desc)
  g = geweke.Geweke()
  for i in range(N):
    prior_sampler.UpdateRelationsFromPrior(s)
    prior_sampler.UpdateFactsFromPrior(s)
    prior_sampler.UpdateRowsFromPrior(s)
    prior_sampler.UpdateObservationsFromPrior(s)
    if i%10==0:
      g._prior_samples.append(copy.deepcopy(s))
  for i in range(N):
    prior_sampler.UpdateRelationsFromPrior(s)
    posterior_sampler.UpdateFacts(s)
    posterior_sampler.UpdateRows(s)
    prior_sampler.UpdateObservationsFromPrior(s)
    if i%10==0:
      g._geweke_samples.append(copy.deepcopy(s))
  return g

if __name__ == "__main__":
  logging.info('Performing test')