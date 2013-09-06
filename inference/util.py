"""Miscellaneous utility functions used in different modules.
"""

from __future__ import division

import difflib
import logging

from IPython import parallel
import matplotlib.pyplot as plt
import numpy as np
import pandas
import argparse
import os


def Dview():
  client = parallel.Client()
  return client[:]


def Densify(df, index):
  """Convert a sparse dataframe into a dense representation.

  Given a dataframe *df* whose index is a subset of *index*,
  fill in the missing entries specified by index-df.index with 0
  """
  d = pandas.Series(np.zeros(len(index)), index=index)
  return d.add(df, fill_value=0.)


class TablesException(Exception):
  pass


# Functions for converting between standard deviation, variance,
# and precision for 1D data
# Handy since normal-gamma distribution parameterized by precision.

def SDToTau(sd):
  return 1./(sd * sd)


def TauToSD(tau):
  return np.sqrt(1./tau)


def SDToVar(sd):
  return sd ** 2


def VarToSD(var):
  return np.sqrt(var)


def VarToTau(var):
  return 1./var


def TauToVar(tau):
  return 1./tau


def PcolorDiscrete(X, ylabels=None, xlabels=None, **kwargs):
  """Produces a pseudocolor plot of a matrix 'X', with optional axes labels.
  """
  if not xlabels:
    xlabels = np.arange(X.shape[1])
  if not ylabels:
    ylabels = np.arange(X.shape[0])
  plt.pcolor(X, cmap=plt.cm.hot, **kwargs)
  plt.colorbar()
  plt.yticks(np.arange(X.shape[0]) + .5, ylabels)
  plt.xticks(np.arange(X.shape[1]) + .5, xlabels)
  ax = plt.gca()
  ax.invert_yaxis()


def VisualizeWeights(w, is_log=True, print_raw=False,
                     transpose=False, **kwargs):
  """ Show a weight matrix as a heatmap.
  Visualize a probabiility matrix *w* as a heatmap where
  w[i,j] = ln P(X=j|Y=i) for two discrete RVs X, Y
  """
  if is_log:
    normalizer = np.logaddexp.reduce(w, 1).reshape((-1, 1))
    w = np.exp(w - normalizer)
  normalizer = sum(w, 1).reshape([-1, 1])
  w /= normalizer
  if transpose:
    w = w.T
  if print_raw:
    x = [['%.2f' % _ for _ in row] for row in w]
    for row_idx, row in enumerate(x):
      print 'Row %d: ' % row_idx, ', '.join(row)
  PcolorDiscrete(w, vmin=0, vmax=1, **kwargs)


def PivotMultiIndex(df, indices, col):
  """
  Just like the Pandas pivoting function, but allows the index to
  correspond to more than one column from the original dataframe.
  """
  a = df.set_index(indices)
  b = a.pivot(index=a.index.values, columns=col)
  b.index = pandas.MultiIndex.from_tuples(b.index)
  b.index.names = indices
  return b


def ConstructNPIndex(index):
  """Convert a Pandas index into a scalar Numpy index.

  Pivots the multi-index of a pandas DataFrame into a matrix X
  such that for a tuple t, df.loc[t]==df.iloc[X[t]].

  Useful for efficiently indexing a dataframe from cython,
   where iloc is efficient and loc is slow.
  """
  if isinstance(index, pandas.DataFrame):
    index = index.index
  v = index.values
  dims = [max(_)+1 for _ in index.labels]
  x = np.empty(dims, int)
  x[:] = -1
  for i in range(len(v)):
    x[v[i]] = i
  return x


def StringSim(s1, s2):
  s1 = s1.lower()
  s2 = s2.lower()
  seq = difflib.SequenceMatcher(a=s1, b=s2)
  return seq.ratio()


def GetRagged(x, y, values, counts):
  n = counts[x, y]
  return values[x, y, :n]


def GetFlags(use_command_line=False):
  defaults = dict(data_dir='~/damastes',  dataset='botany', trace_id=0)
  if use_command_line:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=defaults['data_dir'])
    parser.add_argument('--dataset', '-d', type=str, default=defaults['dataset'])
    parser.add_argument('--trace_id', '-t', type=int, default=defaults['trace_id'])
    try:
      args = parser.parse_args()
    except SystemExit:
      logging.warning('Failed to parse command-line input')
      return GetFlags(False)
    data_dir = args.data_dir
    dataset = args.dataset
    trace_id = args.trace_id
  else:
    data_dir = defaults['data_dir']
    dataset = defaults['dataset']
    trace_id = defaults['trace_id']
  data_dir = os.path.expanduser(data_dir)
  return dict(data_dir=data_dir, dataset=dataset, trace_id=trace_id)


def ConfigureLogging():
  logging.basicConfig(level=logging.DEBUG,
                      format=('%(levelname)s %(asctime)s in '
                              '%(module)s %(lineno)d: %(message)s'))


