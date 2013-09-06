"""Support for Markov Chain Monte Carlo methods.
"""
import cPickle
import itertools
import logging
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas

import posterior_sampler
import util


class Trace(object):
  """Represents a set of samples from one MCMC run.

  Attributes:
    samples: A list of samples from the course of the run,
      which are State objects.
    iters: A list of the iteration number of each stored sample
    time: The time in seconds when each sample was added to the trace
    skip: The thinning interval. Store only every 'skip' iterations.
  """

  def __init__(self):
    self.samples = []
    self.iters = []
    self.time = []
    self.skip = 1

  def ComputeMarginalStatistic(self, f):
    """Compute a marginal statistic.

    Args:
      f: A function which takes a state and returns a statistic

    Returns:
      A list of statistics, one entry per stored sample.
    """
    return [f(_) for _ in self.samples]

  def PlotTrace(self, f, **kwargs):
    """Plot a trace of a statistic in the form of statistic vs. iteration.

    Args:
      f: Function which maps states onto statistics.
      **kwargs: Passed to the line plotting function.
    """
    values = pandas.Series(self.ComputeMarginalStatistic(f), index=self.iters)
    values.plot(**kwargs)

  def PlotHist(self, f, **kwargs):
    """Plot the marginal distribution of a statistic.

    Args:
      f: Function maps states onto statistics
      **kwargs: Passed into histogram-plotting function
    """
    values = self.ComputeMarginalStatistic(f)
    plt.hist(values, **kwargs)

  def AddSample(self, s, iter_no=None):
    self.samples.append(s.Copy())
    self.time.append(time.time())
    self.iters.append(iter_no)

  def Finalize(self):
    self.iters = pandas.Series(self.iters, name='Iterations')
    self.time = pandas.Series(self.time, name='Time (s)')-self.time[0]

  def Last(self):
    """Returns the last state in the trace."""
    return self.samples[-1]

  def Save(self, trace_id):
    """Saves the trace to disk."""
    flags = util.GetFlags()
    filename = os.path.join(flags['data_dir'], 'output', 'trace_%d.pickle' % trace_id)
    with open(filename, 'wb') as f:
      cPickle.dump(self, f, 2)


def LoadTrace(filename):
  """Load a trace from disk."""
  with open(filename, 'rb') as f:
    trace = cPickle.load(f)
  return trace


def Mcmc(intial_state, n_iters=100, max_time=np.inf, skip=10):
  """Runs MCMC until some termination condition is reached.

  Args:
    initial_state: The starting state of the chain. A State object.
    n_iters: The number of iterations to run for.
    max_time: The maximum number of seconds to run for.
    skip: The thinning interval.

  Returns:
    A Trace object
  """
  s = intial_state
  trace = Trace()
  start_time = time.time()
  # itertools.count is used here to allow the possiblity that n_iters if infinity,
  # in which case termination is based only on max_time or a keyboard interrupt.
  for n in itertools.count():
    if n >= n_iters:
      break
    if n % skip == 0:
      trace.AddSample(s, n)
      logging.info('On iteration %r', n)
    if time.time() - start_time > max_time:
      break
    try:
      posterior_sampler.UpdateState(s)
    except KeyboardInterrupt:  # Terminate inference early by pressing Ctrl-C
      logging.info('Keyboard interrupt')
      break
  trace.Finalize()
  trace.Last().Plot()
  return trace

