"""Cythonized code for computing CPU-intensive posterior updates.

In the complexity analyze, variables are as follows:
n_subs: Number of active entities (currently = max # of entities)
n_rels: NUmber of active relations
n_tokens: Maximum number of tokens
n_topics: Maximum number of topics
n_ranges: Maximum number of ranges
n_cells: Total number of table cells in the dataset
n_fact_usage: The typical number of each cell is associated with a randomly-
chosen fact. In the worst case, same as n_cells.
n_rel_usage: Typical number of columns assigned to each relation
n_sub_usage: Typical number of subjects assigned to each subject


"""

#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=True
#cython: infer_types=True
#cython: embedsignature=True
#cython: cdivision=True


from __future__ import division

import numpy as np

import cCollect
import cDists
#import google3.knowledge.graph.vault.open_world.inference.cDists
# TODO(malmaud): Get cImports working with blaze. Vital for efficiency.
#cimport cDists
from libc.math cimport exp, log, sqrt

cdef double inf=np.inf


cpdef double SumDoubleArray(double[:] x):
  cdef double s=0.
  cdef long N=x.shape[0]
  cdef long n

  for n in range(N):
    s += x[n]
  return s


cpdef long SumLongArray(long[:] x):
  cdef long s=0
  cdef long N=x.shape[0]
  cdef long n

  for n in range(N):
    s += x[n]
  return s

# TODO(malmaud): Improve documentation by adding complexity analysis to each docstring
# TODO(malmaud): Link to references explaining the update equations in each docstring

cpdef double CalculateEntityTokenLogLikelihood(long zs, long xs, double alpha,
                                               long[:, :] counts):
  """ Calculates the probability of a given token for a given entity.
  P(xs|zs), where P(*|zs)~Dir(alpha) and counts[x,y]=
  of times the token 'y' occured for subject 'x'

  Takes time O(n_token).
  """
  cdef long n_tokens
  cdef double num, denom

  n_tokens = counts.shape[1]
  num = alpha + counts[zs, xs]
  # TODO(malmaud): Cache this sum of the # of references to each entity
  # Can probably make this constant time
  denom = alpha * n_tokens + SumLongArray(counts[zs])
  return log(num) - log(denom)


cpdef long IsInArray(long probe, long[:] x):
  cdef long n, N

  N = x.shape[0]
  for n in range(N):
    if x[n] == probe:
      return 1
  return 0


def UpdateRowPosterior(state):
  """Performs a posterior update of the subject indicators for each row.

  Complexity: If conditioning on the facts table, O(n_cells * n_subs)
  If marginalizing over facts, O(n_cells * n_subs^2 * n_fact_usage)
  """
  # TODO(malmaud): Get a speed increase by ignoring latents attached to non-active relations
  cdef long[:, :] subject_token_counts, data_dims
  cdef long[:, :] rows_index, cols_index, facts_index
  cdef long[:, :, :] cells_index
  cdef long[:] subject_counts, new_zs_array, cols_type, cols_zr
  cdef long[:] facts_ref, cells_xo_ref, rows_xs, subs_in_table
  cdef double alpha_token, alpha_sub, llh_header
  cdef double llh_prior, sd_xo, zo_real, xo_real
  cdef double[:] llh_col, facts_real, cells_xo_real
  cdef double[:, :] posterior
  cdef long table, row, zs, col, col_type
  cdef long n_cols, n_rows, n_tables, row_index, zr, n_subs
  cdef long fact_index, col_index, cell_index, zo, xo, new_zs, old_zs, xs
  #cdef cDists.normal_parms numeric_parms


  subject_token_counts = cCollect.TabulateSubjectTokenCounts(state, include_cells=True)
  subject_counts = cCollect.TabulateTopicSubjectCounts(state)[0]
  alpha_token = state.hypers['alpha_token']
  alpha_sub = state.hypers['alpha_sub']
  posterior = np.empty((len(state.rows), state.n_subs))
  new_zs_array = -1 * np.empty(len(state.rows), np.int)
  n_tables = state.n_tables
  data_dims = state.data_dims
  rows_index = state.rows_index
  cols_index = state.cols_index
  facts_index = state.facts_index
  cells_index = state.cells_index
  rows_zs = state.rows['zs'].values
  rows_xs = state.rows['xs'].values
  cols_type = state.cols['type'].values
  cols_zr = state.cols['zr'].values
  facts_ref = state.facts['ref'].values
  cells_xo_ref = state.cells['xo_ref'].values
  facts_real = state.facts['real'].values
  cells_xo_real = state.cells['xo_real'].values
  sd_xo = state.hypers['sd_xo']
  n_subs = state.n_subs
  numeric_parms = cDists.normal_parms()

  for table in range(state.n_tables):
    n_rows = data_dims[table, 0]
    n_cols = data_dims[table, 1]
    llh_col = np.empty(n_cols)
    subs_in_table = np.repeat(-1, n_rows)
    for row in range(n_rows):
      row_index = rows_index[table, row]
      old_zs = rows_zs[row_index]
      subject_counts[old_zs] -= 1
      xs = rows_xs[row_index]
      subject_token_counts[old_zs, xs] -= 1
      for zs in range(n_subs):
        llh_header = CalculateEntityTokenLogLikelihood(zs, xs, alpha_token,
                                                       subject_token_counts)
        llh_prior = log(subject_counts[zs] + alpha_sub)

        # Do not allow a subject to be duplicated within a table.
        if IsInArray(zs, subs_in_table):
          llh_prior = -inf
        for col in range(n_cols):
          col_index = cols_index[table, col]
          col_type = cols_type[col_index]
          zr = cols_zr[col_index]
          fact_index = facts_index[zr, zs]
          cell_index = cells_index[table, row, col]
          if col_type == 0:
            zo = facts_ref[fact_index]
            xo = cells_xo_ref[cell_index]
            llh_col[col] = CalculateEntityTokenLogLikelihood(zo, xo,
                                                             alpha_token,
                                                             subject_token_counts)
          elif col_type == 1:
            zo_real = facts_real[fact_index]
            xo_real = cells_xo_real[cell_index]
            numeric_parms.mu = zo_real
            numeric_parms.sd = sd_xo
            llh_col[col] = cDists.NormalLike(xo_real, numeric_parms)
        # The posterior of a row being assigned to a given subject is the sum
        # of three terms: The prior, based on the topic of the table;
        # the likelihood of the string in the row's subject column (llh_header),
        # and the likelihood of the cells in that row, were that row to be
        # assigned to the given subject.
        posterior[row_index, zs] = (llh_prior + llh_header
                                    + SumDoubleArray(llh_col))
      new_zs = cDists.SampleCategoricalLn(posterior[row_index])
      subs_in_table[row] = new_zs
      subject_token_counts[new_zs, xs] += 1
      subject_counts[new_zs] += 1
      new_zs_array[row_index] = new_zs
  state.rows.zs = new_zs_array


def CalculateEntityRangesPosterior(s):
  """Calculates the posterior distribution of the range of each relation.

  Fact table is marginalized out.

  Complexity is O(n_rels * n_ranges * n_subs * n_rel_usage)
  """
  cdef double[:, :] posterior
  cdef long n_subs, zs, zt, n_topics, n_rels, zr, i
  cdef long topic, zo, range_type, n_types
  cdef double y, alpha
  cdef double[:, :] sub_topic_w, topic_sub_w, sub_token_w, type_sub_w
  cdef long[:, :] xo_array
  cdef long[:] counts, topic_counts, range_ref, xo, type_counts

  n_subs = s.n_subs
  n_types = s.n_types
  n_rels = s.n_rels

  posterior = np.zeros((n_rels, n_types))
  sub_token_w = np.log(s.sub_token_weights)
  type_sub_w = np.log(s.type_sub_weights)
  observed_values = cCollect.ExtractObservedValuesForEachRelation(s)
  counts = observed_values['counter']
  xo_array = observed_values['refs']
  type_counts = cCollect.TabulateRangeCounts(s)
  range_ref = s.rels.range_ref.values
  alpha = s.hypers['alpha']
  active_array = s.rels.active.values
  for zr in range(n_rels):
    type_counts[range_ref[zr]] -= 1
    for range_type in range(n_types):
      posterior[zr, range_type] += log(alpha + type_counts[range_type])
      xo = xo_array[zr, :counts[zr]]
      for i in range(xo.shape[0]):
        y = 0.
        for zo in range(n_subs):
          y += exp(sub_token_w[zo, xo[i]] + type_sub_w[range_type, zo])
          # TODO(malmaud): replace with LogSumExp
        posterior[zr, range_type] += log(y)
    range_ref[zr] = cDists.SampleCategoricalLn(posterior[zr, :])
    type_counts[range_ref[zr]] += 1
  s.rels.range_ref = range_ref
  return np.asarray(posterior)


def UpdateNumericRangesPosterior(s):
  """Resamples the mean and standard deviation of each relation's range.

  Based on the value of all facts associated with the relation and the
  global Normal-Gamma prior over the range of numeric relations.

  Conditioned on facts, complexity is O(n_rels * n_rels_usage)

  """
  cdef double[:] mu, sd, x
  cdef long zr, n_rels
  #cdef cDists.normal_gamma_parms prior
  #cdef cDists.normal_parms new_range
  cdef long[:] counts

  d = cCollect.ExtractFactsForEachRelation(s)
  counts = d['counter']
  zo_array = d['facts']
  n_rels = s.n_rels
  prior = s.hypers['rel_range']
  mu = np.empty(n_rels)
  sd = np.empty(n_rels)
  active_array = s.rels.active.values
  for zr in range(n_rels):
    x = zo_array[zr, :counts[zr]]
    UpdateNumericRelation(s, zr, x, mu, sd)
  s.rels.range_real_mu = mu
  s.rels.range_real_sd = sd


def CalculateEntityFactsPosterior(s):
  """Calculate the posterior distribution of entity-valued facts.

  Conditioned on facts, complexity is O(n_subs * n_cells)
  """
  cdef long n_facts, n_cells, n_subs, n_rels, n, zr
  cdef long zt, zs, zo, xo, fact_idx, cell_type
  cdef double[:, :] posterior
  cdef long[:] zr_array, zs_array, xo_array, topic_array
  cdef long[:] cell_type_array, active_array, type_array
  cdef double[:,:] sub_token_w, topic_sub_w, type_sub_w
  cdef long[:,:] facts_index

  n_facts = len(s.facts)
  n_cells = len(s.cells)
  n_subs = s.n_subs
  n_rels = s.n_rels

  posterior = np.zeros((n_facts, n_subs))
  cell_facts = cCollect.JoinCellsWithFactsTable(s)
  zr_array = cell_facts['zr']
  zs_array = cell_facts['zs']
  cell_type_array = cell_facts['type']
  xo_array = s.cells.xo_ref.values
  sub_token_w = np.log(s.sub_token_weights)
  type_sub_w = np.log(s.type_sub_weights)
  type_array = s.rels.range_ref.values
  facts_index = s.facts_index
  active_array = s.rels.active.values
  for n in range(n_cells):
    zr = zr_array[n]
    zs = zs_array[n]
    xo = xo_array[n]
    cell_type = cell_type_array[n]
    if cell_type!=0:
      continue
    fact_idx = facts_index[zr, zs]
    for zo in range(n_subs):
      posterior[fact_idx, zo] += sub_token_w[zo, xo]  #TODO(malmaud): collapse
  for zr in range(n_rels):
    if not active_array[zr]:
      continue
    zt = type_array[zr]
    for zs in range(n_subs):
      fact_idx = facts_index[zr, zs]
      for zo in range(n_subs):
        posterior[fact_idx, zo] += type_sub_w[zt, zo]  #TODO(malmaud): collapse
  return np.asarray(posterior)


def UpdateTableTopics(s):
  """Update the latent topic associated with each table.

  Based on the global prior over topics and the the distribution of
  subjects and relations in each table.
  """
  cdef long n_topics, n_tables, i, topic, zs, zr, row, col
  cdef long[:] zt, counts
  cdef double alpha
  cdef double[:] w
  cdef long[:, :] data_dims, row_index, col_index
  cdef long[:] zs_array, xr_rray, zr_array
  cdef double[:, :] topic_sub_w, topic_rel_w
  zt = s.tables.zt.values
  n_topics = s.n_topics
  n_tables = len(zt)
  alpha = s.hypers['alpha']
  counts = cCollect.TabulateTopicCounts(s)
  w = np.empty(n_topics)
  zs_array = s.rows.zs.values
  zr_array = s.cols.zr.values
  row_index = s.rows_index
  col_index = s.cols_index
  data_dims = s.data_dims
  topic_sub_w = np.log(s.topic_sub_weights)
  topic_rel_w = np.log(s.topic_rel_weights)
  alpha_rel = np.repeat(alpha, s.n_rels)
  rel_counts = cCollect.TabulateTopicRelationCounts(s)
  for i in range(n_tables):
    counts[zt[i]] -= 1

    for topic in range(n_topics):
      w[topic] = log(counts[topic] + alpha)
      for row in range(data_dims[i, 0]):
        zs = zs_array[row_index[i, row]]
        w[topic] += topic_sub_w[topic, zs]

      for col in range(data_dims[i, 1]):
        zr = zr_array[col_index[i, col]]
        w[topic] += topic_rel_w[topic, zr]
    zt[i] = cDists.SampleCategoricalLn(w)

    counts[zt[i]]  += 1
  s.tables.zt = zt


def TabulateFactPosteriorStats(state):
  """Calculate sufficient statistics for updating numeric facts.

  For each latent fact, tabulates the sum of all observed numeric cells
  currently assigned to that fact. Also tabulates the number of such cells.
  These statistics are sufficient for calculating the posterior over each
  numeric fact.

  Returns:
    A dictionary containing the sums and counts statistics, with one entry
    per fact.

  Complexity is O(n_cells)
  """
  cdef long n_subs, n_rels, n, N, index, zr, zs
  cdef double[:] sums, real_array
  cdef long[:]  zs_array, zr_array, type_array
  cdef double xo
  cdef long[:, :] fact_index
  n_subs = state.n_subs
  n_rels = state.n_rels
  sums = np.zeros((n_rels * n_subs))
  lens = np.zeros((n_rels * n_subs), np.int)
  cells = cCollect.JoinCellsWithFactsTable(state)
  zs_array = cells['zs']
  zr_array = cells['zr']
  real_array = state.cells.xo_real.values
  type_array = cells['type']
  fact_index = state.facts_index
  N = zs_array.shape[0]
  for n in range(N):
    if type_array[n]==1:
      zs=zs_array[n]
      zr=zr_array[n]
      index = fact_index[zr, zs]
      xo=real_array[n]
      sums[index] += xo
      lens[index] += 1
  return dict(sums=np.asarray(sums), lens=np.asarray(lens))


cpdef long FindFirstFalse(long[:] x):
  cdef long N=x.shape[0]
  cdef long n
  for n in range(N):
    if x[n]==0:
      return n
  return -1


cpdef UpdateNumericRelation(s, long zr, double[:] numeric_facts, double[:] mu, double[:] sd):
  """Sample the mean and standard deviation of a given relation.

  Args:
    zr: ID of the relation to update
    numeric_facts: An array of numeric facts assigned to the given relation.
    mu: Array to store the resampled mean to.
    sd: Array to store the resampled standard deviation to.

  Complexity is O(|numeric_facts|)
  """
  prior_parms = s.hypers['rel_range']
  updated_parms = cDists.NormalGammaPosteriorParms(prior_parms, numeric_facts)
  new_range = cDists.SampleNormalGamma(updated_parms)
  mu[zr] = new_range.mu
  sd[zr] = new_range.sd


cpdef SampleNumericRelationPrior(state):
  """Samples the mean and SD of a relation from the prior."""
  prior_parms = state.hypers['rel_range']
  new_range = cDists.SampleNormalGamma(prior_parms)
  return new_range


cpdef UpdateNumericRelationFromPrior(state, long zr, double[:] mu, double[:] sd):
  new_range = SampleNumericRelationPrior(state)
  mu[zr] = new_range.mu
  sd[zr] = new_range.sd


cpdef UpdateEntityRelation(state, long zr, long[:] x, long[:] rel_range):
  """
  Update the domain of relation 'zr'
  """
  cdef double alpha_token
  cdef long[:, :] sub_token_counts
  cdef long n_z, n_x, n_types
  cdef double[:] posterior, type_weights
  cdef double[:, :] type_sub_weights, sub_token_weights

  n_types = state.n_types
  posterior = np.zeros(n_types)
  type_sub_weights = state.type_sub_weights
  type_weights = state.type_weights
  sub_token_weights = state.sub_token_weights
  n_x = x.shape[0]
  n_z = state.n_subs
  sub_token_counts = cCollect.TabulateSubjectTokenCounts(state)
  alpha_token = state.hypers['alpha_token']
  for zt in range(n_types):
    posterior[zt] += CalculateEntityColumnLogLikelihood(state, x, None, -1, zt,
                                                        None, None,
                                                        sub_token_counts,
                                                        alpha_token)
    posterior[zt] += log(type_weights[zt])
  rel_range[zr] = cDists.SampleCategoricalLn(posterior)


cpdef UpdateEntityRelationFromPrior(s, long zr, long[:] rel_range):
  new_type = SampleEntityRelationFromPrior(s)
  rel_range[zr] = new_type


cpdef SampleEntityRelationFromPrior(s):
  return cDists.SampleCategorical(s.type_weights)


cpdef CalculateRealColumnLogLikelihood(double[:] observed_numeric_cells,
                                       double[:, :, :] x_prev,
                                       long[:, :] x_prev_counts,
                                       long[:] zs,
                                       long zr, double mu_range,
                                       double range_real_sd,
                                       double sd_xo):
  """Calculates the likelihood of a column.
   Based on the numeric values of the cells that appear in it, the global
   prior on the range of numeric relations, and the set of other numeric literals
   which appear in different columns of the relation.
   """
  cdef double llh, sd_noise
  cdef long n_rows, row
  cdef double[:] empty_array, x_prev_row
  llh = 0.
  sd_noise = sd_xo
  n_rows = len(observed_numeric_cells)
  empty_array = np.empty(0)
  for row in range(n_rows):
    if zr==-1:
      x_prev_row = empty_array
    else:
      x_prev_row = x_prev[zr, zs[row], :x_prev_counts[zr, zs[row]]]
    llh += cDists.NormalPredictiveLogLikelihood(observed_numeric_cells[row],
                                                mu_range,
                                                range_real_sd,
                                                sd_noise,
                                                x_prev_row)
  return llh


cpdef double CalculateEntityColumnLogLikelihood(state, long[:] tokens,
                                                long[:] zs, long zr,
                                                long range_of_column,
                                                long[:] facts,
                                                long[:, :] facts_index,
                                                long[:, :] subject_token_counts,
                                                double alpha_token):
  """Calculates the likelihood of a reference-valued column belonging to a certain relation,
   given the tokens that appear in it.

   Args:
     zr: The latent relation to update. -1 indicates a request for the probability
     of a new relation, marginalized over the prior over relation ranges.

  Conditioned on facts table, takes O(n_rows_in_column * n_subs)
  For a new relation, complexity goes up to O(n_rows_in_column * n_subs^2)
  """
  cdef double llh, zo_llh
  cdef long n_rows, n_subs, row, zo
  cdef double[:, :] sub_token_weights, type_sub_weights
  cdef double[:] row_llh
  llh = 0.
  n_rows = tokens.shape[0]
  n_subs = state.n_subs
  sub_token_weights = state.sub_token_weights
  type_sub_weights = state.type_sub_weights
  if zr == -1:  # Marginal probablity of a new relation
    row_llh = np.zeros(n_subs)
    for row in range(n_rows):
      for zo in range(n_subs):  # assumes zs does not appear multiple times within a table
        zo_llh = 0.
        zo_llh += CalculateEntityTokenLogLikelihood(zo, tokens[row],
                                                    alpha_token, subject_token_counts)
        zo_llh += log(type_sub_weights[range_of_column, zo])
        row_llh[zo] = zo_llh
      llh += cDists.LogSumExpArray(row_llh)
  else:
    for row in range(n_rows):
      zo = facts[facts_index[zr, zs[row]]]  # For expediency, current version uses the sampled facts
      # TODO(malmaud): Go back to marginalizing out facts here
      llh += CalculateEntityTokenLogLikelihood(zo, tokens[row],
                                               alpha_token, subject_token_counts)
  return llh


cpdef Find(long[:] x):
  cdef long n, N, count, idx
  cdef long[:]  y
  count = 0
  N = x.shape[0]
  for n in range(N):
    if x[n]:
      count += 1
  y = np.empty(count, np.int)
  idx = 0
  for n in range(N):
    if x[n]:
      y[idx] = n
      idx += 1
  return np.asarray(y)


def UpdateColumnPosterior(state):
  """Update the relation indicator for each column.
  """
  # TODO(malmaud): Document run-time complexity.
  # TODO(malmaud): Refactor this into shorter, more modular functions
  cdef long n_rels, n_tables, n_tokens, table, col, row, zt
  cdef long cell_type, zr, zr_idx, n_active, n_rows, n_cols
  cdef long col_idx,  xr, new_zr, zr_initial, cell_idx
  cdef long[:, :] data_dims, topic_rel_counts
  cdef long[:, :] cols_index, facts_index, rel_token_counts
  cdef long[:] zr_array, active_array, xr_array
  cdef long[:] zt_array, xo_ref_array, x_ref, zs
  cdef long[:] cells_zo_ref, cells_type, facts, zr_active, rels_in_table
  cdef double[:] mu_zr_array, sd_zr_array, xo_array, cells_zo, posterior, x_real
  cdef double[:, :, :] cells_real
  cdef long[:, :, :] cells_index
  cdef double[:, :] rel_token_weights
  cdef double alpha, alpha_token, llh, mu_zr, sd_zr, obs_noise
  cdef long[:, :] cells_real_count, cells_ref_count
  #cdef cDists.normal_parms new_range

  obs_noise = state.hypers['sd_xo']
  n_rels = state.n_rels
  n_tables = state.n_tables
  n_tokens = state.n_tokens
  data_dims = state.data_dims
  topic_rel_counts = cCollect.TabulateTopicRelationCounts(state)
  cols_index = state.cols_index
  cells_index = state.cells_index
  zr_array = state.cols.zr.values
  active_array = state.rels.active
  xr_array = state.cols.xr.values
  zt_array = state.tables.zt.values
  mu_zr_array = state.rels.range_real_mu.values
  sd_zr_array = state.rels.range_real_sd.values
  topic_zr_array = state.rels.range_ref.values
  xo_array = state.cells.xo_real.values
  xo_ref_array = state.cells.xo_ref.values
  joined_cells = cCollect.JoinCellsWithFactsTable(state)
  cells_zs = joined_cells['zs']
  cells_zo = joined_cells['real']
  cells_zo_ref = joined_cells['ref']
  cells_type = state.cols.type.values
  facts = state.facts.ref.values
  facts_index = state.facts_index
  rel_token_weights = state.rel_token_weights
  posterior = np.empty(n_rels)
  alpha = state.hypers['alpha_zr']
  alpha_token = state.hypers['alpha_token']
  rel_token_counts = cCollect.TabulateRelationTokenCounts(state)

  for table in range(n_tables):
    n_rows, n_cols = data_dims[table, 0], data_dims[table, 1]
    zt = zt_array[table]
    zo_real_array = np.empty(n_rows)
    zo_ref_array = np.empty(n_rows, np.int)
    rels_in_table = np.repeat(-1, n_cols)
    for col in range(n_cols):
      cells_by_fact = cCollect.ExtractObservedValuesForEachFact(state, table, col)
      cells_real = cells_by_fact['reals']
      cells_ref = cells_by_fact['refs']
      cells_real_count = cells_by_fact['real_counter']
      cells_ref_count = cells_by_fact['ref_counter']
      sub_token_counts = (
          cCollect.TabulateSubjectTokenCounts(state,
                                              exclude_table=table,
                                              exclude_column=col))
      col_idx = cols_index[table, col]
      cell_type = cells_type[col_idx]
      zr_initial = zr_array[col_idx]
      xr = xr_array[col_idx]
      topic_rel_counts[zt, zr_initial] -= 1
      if topic_rel_counts[zt, zr_initial] == 0:
        active_array[zr_initial] = 0 #todo only works for one topic
      rel_token_counts[zr_initial, xr] -= 1
      zr_active = Find(active_array)
      n_active = len(zr_active)      
      x_real = np.empty(n_rows)
      x_ref = np.empty(n_rows, np.int)
      zs = np.empty(n_rows, np.int)

      for row in range(n_rows):
        cell_idx = cells_index[table, row, col]
        zs[row] = cells_zs[cell_idx]
        x_real[row] = xo_array[cell_idx]
        x_ref[row] = xo_ref_array[cell_idx]
        zo_real_array[row] = cells_zo[cell_idx]
        zo_ref_array[row] = cells_zo_ref[cell_idx]

      for zr_idx in range(n_active + 1):
        if cell_type==0:
          if zr_idx < n_active:
            zr = zr_active[zr_idx]
            zt_zr = topic_zr_array[zr]
          else:
            zr = -1
            zt_zr = SampleEntityRelationFromPrior(state)
          llh = (
              CalculateEntityColumnLogLikelihood(state, x_ref,  zs, zr, zt_zr,
                                                 facts, facts_index,
                                                 sub_token_counts, alpha_token))
        elif cell_type==1:
          if zr_idx < n_active:
            zr = zr_active[zr_idx]
            mu_zr = mu_zr_array[zr]
            sd_zr = sd_zr_array[zr]
          else:
            zr = -1
            new_range = SampleNumericRelationPrior(state)
            mu_zr = new_range.mu
            sd_zr = new_range.sd
          llh = CalculateRealColumnLogLikelihood(x_real, cells_real,
                                                 cells_real_count,
                                                 zs, zr, mu_zr, sd_zr, obs_noise)

        if zr_idx < n_active:
          prior = log(topic_rel_counts[zt, zr])

          #  Do not allow the relation to be duplicated within a table
          if IsInArray(zr_active[zr_idx], rels_in_table):
            prior = -inf
          p_header = CalculateEntityTokenLogLikelihood(zr, xr, alpha_token,
                                                       rel_token_counts)
        else:
          prior = log(alpha)
          p_header = -log(n_tokens)
        posterior[zr_idx] = prior + p_header + llh
      new_zr_idx = cDists.SampleCategoricalLn(posterior[:n_active+1])
      if new_zr_idx < n_active:
        new_zr = zr_active[new_zr_idx]
      else:
        new_zr = FindFirstFalse(active_array)
        active_array[new_zr] = 1
        if cell_type==0:
          UpdateEntityRelation(state, new_zr, x_ref, topic_zr_array)
          UpdateNumericRelationFromPrior(state, new_zr, mu_zr_array, sd_zr_array)
        elif cell_type==1:
          UpdateNumericRelation(state, new_zr, x_real, mu_zr_array, sd_zr_array)
          # todo: update based on x_real directly instead of zo_real_array
          UpdateEntityRelationFromPrior(state, new_zr, topic_zr_array)
      rels_in_table[col] = new_zr
      zr_array[col_idx] = new_zr
      topic_rel_counts[zt, new_zr] += 1
      rel_token_counts[new_zr, xr] += 1
  state.cols.zr = zr_array
  state.rels.range_real_mu = mu_zr_array
  state.rels.range_real_sd = sd_zr_array
  state.rels.range_ref = topic_zr_array

