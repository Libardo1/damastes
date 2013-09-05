"""Perform Geweke testing to verify correctness of inference.

Based on "Getting it right",
Geweke (2012), Journal of the American Statistical Association.

See http://amstat.tandfonline.com/doi/abs/10.1198/016214504000001132#.UiAtW2R4ajQ.
"""

import logging
import matplotlib.pyplot as plt
import posterior_sampler
import prior_sampler


class Geweke(object):
  """Stores the results of a Geweke test.

  Attributes:
    prior_samples: A list of samples drawn from the prior.
    geweke_samples: A list of samples created from posterior inference.
  """

  def __init__(self):
    self._prior_samples = []
    self._geweke_samples = []

  def PlotStatistic(self, statistic):
    """Compare a marginal statistic across the prior samples and geweke samples.

    Args:
      statistic: A function which maps states into scalar statistics.
    """
    f_prior = [statistic(s) for s in self._prior_samples]
    f_geweke = [statistic(s) for s in self._geweke_samples]
    _, ax = plt.subplots(2, 1, sharex=True)
    plt.sca(ax[0])
    parms = dict(bins=20)
    plt.hist(f_prior, **parms)
    plt.grid()
    plt.title('Ground truth')
    plt.sca(ax[1])
    plt.hist(f_geweke, **parms)
    plt.grid()
    plt.title('Inferred')

  def __str__(self):
    return 'Geweke test with %r samples' % len(self._geweke_samples)

  def Test(self, n_prior, n_posterior, data_dims, update_state=None,
           update_data=None, skip=10):
    """Performs a Geweke test.

    Args:
      n_prior: The number of samples to draw from the prior
      n_posterior: The number of Gibbs steps to perform
      data_dims: The dimensions of the observed data to simulate
      update_state: The transition function for performing one sweep of
        posterior inference. If None, uses posteriors.UpdateState
      update_data: The transition function for resampling the observed
        data, conditioned on the latent state. If None, uses priors.UpdateData
      skip: The thinning interval for storing Geweke samples

    """
    if update_state is None:
      update_state = posterior_sampler.UpdateState
    if update_data is None:
      update_data = prior_sampler.UpdateObservationsAndTypesFromPrior
    prior_samples = [prior_sampler.SamplePrior(data_dims) for _ in range(n_prior)]
    s = prior_sampler.SamplePrior(data_dims)
    trace = []
    for i in range(n_posterior):
      update_state(s)
      update_data(s)
      if i % skip == 0:
        trace.append(s.copy())
        logging.info('geweke test %d', i)
    self._prior_samples.extend(prior_samples)
    self._geweke_samples.extend(trace)

