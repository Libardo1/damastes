#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=True
#cython: infer_types=True
#cython: embedsignature=True
#cython: cdivision=True


from __future__ import division
from libcpp.vector cimport vector
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport exp, log, sqrt
import numpy as np
#cimport numpy as np
cimport gsl

cdef gsl.gsl_rng* _RNG = gsl.gsl_rng_alloc(gsl.gsl_rng_taus)
cdef double _INF = np.inf
ctypedef unsigned int uint


cpdef double SampleUniform():
  return gsl.gsl_rng_uniform(_RNG)


cpdef double SampleGamma(double shape, double rate):
  return gsl.gsl_ran_gamma(_RNG, shape, 1./rate)


cpdef double CalculateDirichletMultinomialLikelihood(long[:] counts,
                                                     double[:] alpha):
  cdef long N_levels, n
  cdef double N, alpha_sum, numerator, denominator, correction_factor, normalizer, net_llh
  N_levels = len(counts)
  N = 0. 
  alpha_sum = 0.
  numerator = 0.
  denominator = 0.
  correction_factor = 0.
  for n in range(N_levels):
    N += counts[n]
    alpha_sum += alpha[n]
    numerator += CalculateGammaLn(counts[n] + alpha[n])
    denominator += CalculateGammaLn(alpha[n])
    correction_factor += CalculateGammaLn(counts[n] + 1)
  correction_factor = CalculateGammaLn(N + 1.) - correction_factor
  normalizer = CalculateGammaLn(alpha_sum) - CalculateGammaLn(N + alpha_sum)
  net_llh = normalizer + numerator - denominator + correction_factor
  return net_llh


cpdef double CalculateGammaLn(double x):
  """Natural logarithm of the gamma function."""
  return gsl.gsl_sf_lngamma(x)


cpdef long SampleCategorical(double[:] weights):
  """Samples from an arbitrary discrete distribution.

  Faster than Numpy's version since it does not perform various safety checks.
  """
  cdef uint N
  cdef long i
  cdef vector[double] cum_sum
  cdef double prev, Z, r
  prev = 0.
  N = weights.shape[0]
  cum_sum.resize(N)
  for i in range(N):
    cum_sum[i] = weights[i] + prev
    prev = cum_sum[i]
  Z = prev
  r = SampleUniform() * Z
  for i in range(N):
    if r <= cum_sum[i]:
      return i
  return -1


cpdef SampleDirichletMultinomial(long[:] assignments, double[:] alpha):
  """Performs an in-place update of categorical variables."""
  cdef uint k, K, n, N
  cdef double[:] counts=np.copy(np.asarray(alpha))

  K = alpha.shape[0]
  N = assignments.shape[0]
  for n in range(N):
    counts[assignments[n]] += 1.
  for n in range(N):
    counts[assignments[n]] -= 1.
    assignments[n] = SampleCategorical(counts)
    counts[assignments[n]] += 1.


def SampleCRPPrior(long N, double alpha):
  cdef long MAX_TABLES = 1000
  cdef long[:] assignments = np.empty(N, np.int)
  cdef long n, new_table
  cdef long num_occupied_tables = 1
  cdef double[:] counts = np.zeros(MAX_TABLES)
  assignments[0] = 0
  counts[0] += 1.
  for n in range(1, N):
    counts[num_occupied_tables] = alpha
    new_table = SampleCategorical(counts[:num_occupied_tables + 1])
    assignments[n] = new_table
    counts[num_occupied_tables] = 0.
    counts[new_table] += 1
    if new_table == num_occupied_tables:
      num_occupied_tables += 1
  return np.asarray(assignments)


cpdef double LogSumExp(double first_term, double second_term):
  cdef double normalizer
  if first_term == -_INF and second_term == -_INF:
    return -_INF
  if first_term > second_term:
    normalizer = first_term
  else:
    normalizer = second_term
  return log(exp(first_term - normalizer) + exp(second_term - normalizer)) + normalizer


cpdef double LogSumExpArray(double[:] x):
  cdef double cumulator
  cdef long n, N
  cumulator = x[0]
  N = x.shape[0]
  for n in range(1, N):
    cumulator = LogSumExp(cumulator, x[n])
  return cumulator


cpdef long SampleCategoricalLn(double[:] weights):
  cdef long N, i
  cdef vector[double] cum_sum
  cdef double previous, normalizer, r
  N = weights.shape[0]
  cum_sum.resize(N)
  cum_sum[0] = weights[0]
  previous = cum_sum[0]
  for i in range(1, N):
    cum_sum[i] = LogSumExp(weights[i], previous)
    previous = cum_sum[i]
  normalizer = previous
  r = log(SampleUniform()) + normalizer
  for i in range(N):
    if r <= cum_sum[i]:
      return i
  return -1


cpdef  SampleCategoricalLnArray(double[:, :] weights):
  cdef long[:] out
  cdef uint n,N
  N=weights.shape[0]
  out=np.empty(N, np.long)
  for n in range(N):
    out[n] = SampleCategoricalLn(weights[n, :])
  return np.asarray(out)


cpdef  SampleCategoricalArray(double[:] weights, int N):
  cdef long[:] out = np.empty(N, np.long)
  cdef uint i
  for i in range(N):
    out[i] = SampleCategorical(weights)
  return np.asarray(out)


cpdef double SampleNormal(double mu, double sd):
  return gsl.gsl_ran_gaussian(_RNG, sd) + mu


cpdef SampleDirichlet(double[:] alpha, double[:] output):
  """Samples a Dirichlet-distributed random variate.

  Stores the sample in the provided 'output' array.
  Assumes the caller has already allocated that array.
  """

  cdef uint N, i
  cdef double normalizer
  N = len(alpha)
  normalizer = 0.
  for i in range(N):
    output[i] = gsl.gsl_ran_gamma(_RNG, alpha[i], 1.)
    normalizer += output[i]
  # A Dirichlet sample can be obtained by
  # normalizing Gamma variates to sum to 1.
  for i in range(N):
    output[i] = output[i]/normalizer


cpdef  SampleDirichletArray(double[:, :] alpha):
  cdef double[:, :] output = np.empty_like(alpha)
  cdef uint n,N
  N = alpha.shape[0]
  for n in range(N):
    SampleDirichlet(alpha[n, :], output[n, :])
  return np.asarray(output)


cpdef double CalculateMultinomialLogLikelihood(long[:] outcomes, double[:] weights):
  cdef double llh = 0.
  cdef uint n,N
  cdef uint i
  N = outcomes.shape[0]
  for n in range(N):
    i = outcomes[n]
    llh += log(weights[i])
  return llh


cdef class normal_parms:

  def __cinit__(self, double mu=0., double sd=1.):
    self.mu = mu
    self.sd = sd

  def __repr__(self):
    s = "Normal (%r, %r)" % (self.mu, self.sd)
    return s


cdef class normal_gamma_parms:

  def __cinit__(self, double mu=0., double nu=1.,
                double shape=1., double rate=1.):
    self.mu = mu
    self.nu = nu
    self.shape = shape
    self.rate = rate

  def __repr__(self):
    s = ("NormalGamma (%r, %r, %r, %r)" %
       (self.mu, self.nu, self.shape, self.rate))
    return s


cpdef normal_parms SampleNormalGamma(normal_gamma_parms parms):
  cdef double prec, mu
  cdef normal_parms out  
  prec = SampleGamma(parms.shape, parms.rate)
  mu = SampleNormal(parms.mu, sqrt(1./(parms.nu*prec)))
  out = normal_parms()
  out.mu = mu
  out.sd = sqrt(1./prec)
  return out


cpdef normal_gamma_parms NormalGammaPosteriorParms(normal_gamma_parms parms0,
                                                   double[:] x):
  cdef normal_gamma_parms parms
  cdef double dev, x_sum, ssq, x_mean
  cdef long N, n
  parms = normal_gamma_parms()
  x_sum = 0.
  ssq = 0.
  N = x.shape[0]
  if N==0:
    return parms0
  for n in range(N):
    x_sum += x[n]
  x_mean = x_sum/N
  for n in range(N):
    ssq += (x[n] - x_mean) * (x[n] - x_mean)
  parms.mu = (parms0.mu * parms0.nu + x_sum) / (parms0.nu + N)
  parms.nu = parms0.nu + N
  parms.shape = parms0.shape + N/2.
  dev = x_mean-parms0.mu
  parms.scale = parms0.scale + .5 * (ssq + parms0.nu*N*dev*dev/(parms0.nu + N))
  return parms


cpdef normal_parms NormalPosteriorParms(double mu0, double sd0,
                                        double sd, double[:] x):
  # TODO(malmaud): enhance this for unknown value of observation noise
  cdef long n, N
  cdef double Z, mu,  total, N_dbl
  cdef normal_parms parms
  parms = normal_parms()
  N = len(x)
  N_dbl = N
  if N==0:
    parms.mu = mu0
    parms.sd = sd0
    return parms
  total = 0.
  for n in range(N):
    total += x[n]
  mu = mu0/(sd0*sd0) + total/(sd * sd)
  Z = 1./(mu0 * mu0) + N_dbl/(sd * sd)
  parms.mu = mu/Z
  parms.sd = sqrt(1./Z)
  return parms


cpdef double NormalLike(double x, normal_parms parms):
  cdef double centered
  cdef double var= parms.sd * parms.sd
  cdef long n, N
  N = 1
  cdef double out=0.
  for n in range(N):
    centered = x - parms.mu
    out += -.5/var * centered * centered
  return out


cpdef double NormalPredictiveLogLikelihood(double x_pred, double mu0,
                                    double sd0, double sd, double[:] x):
  cdef normal_parms parms = NormalPosteriorParms(mu0, sd0, sd, x)
  parms.sd = sqrt(parms.sd * parms.sd + sd0 * sd0)
  return NormalLike(x_pred, parms)


cpdef SetRandomSeed(long seed):
  gsl.gsl_rng_set(_RNG, seed)


