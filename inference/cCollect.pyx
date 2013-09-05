#cython: boundscheck=False
#cython: wraparound=False
#cython: profile=True
#cython: infer_types=True
#cython: embedsignature=True
#cython: cdivision=True

import numpy as np

cdef long MAX_FACTS = 1000  # The maximum number of observed data points for a given relation or subject

def JoinCellsWithFactsTable(state):
  """Return the latent information associated with each cell of each table.

  Args:
    state: An instance of a state object from State.py

  Returns:
    A dictionary of the value of latent variables,
    with one entry per cell in the inputted state.
  """
  cdef long N, row, col, n, zs, zr, table
  cdef double fact
  cdef double[:] real_facts
  cdef long[:] ref_facts, type_facts, zr_facts, zs_facts
  cdef long[:] zs_array, zr_array, cell_rows
  cdef long[:] cell_cols, cell_tables, entity_fact_array, types_array
  cdef double[:] numeric_fact_array
  cdef long[:, :] fact_index, zs_index, zr_index

  N = len(state.cells)
  real_facts=np.empty(N)
  ref_facts = np.empty(N, np.int)
  type_facts = np.empty(N, np.int)
  zs_facts = np.empty(N, np.int)
  zr_facts = np.empty(N, np.int)
  domain = np.empty(N, np.int)
  cell_rows = state.cells.row.values
  cell_cols = state.cells.col.values
  cell_tables = state.cells.table.values
  zs_array = state.rows.zs.values
  zr_array = state.cols.zr.values
  domain_array = state.rels.range_ref
  numeric_fact_array = state.facts.real.values
  entity_fact_array = state.facts.ref.values
  zs_index = state.rows_index
  zr_index = state.cols_index
  fact_index = state.facts_index
  types_array = state.cols.type.values
  for n in range(N):
    row = cell_rows[n]
    col = cell_cols[n]
    table = cell_tables[n]
    zs = zs_array[zs_index[table, row]]
    zr = zr_array[zr_index[table, col]]
    fact = numeric_fact_array[fact_index[zr, zs]]
    real_facts[n] = fact
    ref_facts[n] = entity_fact_array[fact_index[zr, zs]]
    type_facts[n] = types_array[zr_index[table, col]]
    domain[n] = domain_array[zr]
    zs_facts[n] = zs
    zr_facts[n] = zr
  return dict(real=real_facts, ref=ref_facts, type=type_facts,
              zs=zs_facts, zr=zr_facts, domain=domain)


def ExtractObservedValuesForEachFact(state, long exclude_table=-1, long exclude_column=-1):
  """ Given a state, return a dictionary containing all observed values for each (zr, zs),
  plus  a count of how many observed values for each (zr, zs).
  """
  cdef long zs, zr, entity_value, count, n, N, cell_type, n_rels, n_subs
  cdef long[:] zs_array, zr_array, type_array, xo_ref_array
  cdef double[:] xo_real_array
  cdef long[:, :] counter, real_counter
  cdef long[:, :, :] entity_values
  cdef double numeric_value
  cdef double[:, :, :] numeric_values
  cell_facts = JoinCellsWithFactsTable(state)
  n_rels = state.n_rels
  n_subs = state.n_subs
  counter = np.zeros((n_rels, n_subs), np.int)
  entity_values = np.empty((n_rels, n_subs, MAX_FACTS), np.int)
  numeric_values = np.empty((n_rels, n_subs, MAX_FACTS))
  real_counter = np.zeros((n_rels, n_subs), np.int)
  zs_array = cell_facts['zs']
  zr_array = cell_facts['zr']
  type_array = cell_facts['type']
  xo_ref_array = state.cells.xo_ref.values
  xo_real_array = state.cells.xo_real.values
  column_array = state.cells.col.values
  table_array = state.cells.table.values
  N = len(xo_ref_array)
  for n in range(N):
    if (table_array[n] == exclude_table) and (column_array[n] == exclude_column):
      continue
    cell_type = type_array[n]
    zs = zs_array[n]
    zr = zr_array[n]
    entity_value = xo_ref_array[n]
    numeric_value = xo_real_array[n]
    if cell_type==0:
      count = counter[zr, zs]
      entity_values[zr, zs, count] = entity_value
      counter[zr, zs] += 1
    elif cell_type==1:
      count = real_counter[zr, zs]
      numeric_values[zr, zs, count] = numeric_value
      real_counter[zr, zs] += 1
  return dict(refs=np.asarray(entity_values), ref_counter=np.asarray(counter),
              reals=np.asarray(numeric_values), real_counter=np.asarray(real_counter))


def ExtractObservedValuesForEachRelation(state):
  """
  For each zr, returns the list of observed data
  corresponding to that relation
  """
  cdef long[:,:,:] entity_values
  cdef long[:, :] ref_counter, real_counter
  cdef long[:] entity_valued_relation_counter, numeric_valued_relation_counter
  cdef long[:, :] refs_zr
  cdef double[:, :] reals_zr
  cdef double[:, :, :] numeric_values
  cdef long zr, zs, i, n_subs, n_rels
  n_subs = state.n_subs
  n_rels = state.n_rels
  obs_values_per_fact_dict = ExtractObservedValuesForEachFact(state)
  entity_values = obs_values_per_fact_dict['refs']
  ref_counter = obs_values_per_fact_dict['ref_counter']
  numeric_values = obs_values_per_fact_dict['reals']
  real_counter = obs_values_per_fact_dict['real_counter']
  entity_valued_relation_counter = np.zeros(state.n_rels, np.int)
  numeric_valued_relation_counter = np.zeros(state.n_rels, np.int)
  refs_zr =  np.empty((state.n_rels, MAX_FACTS), np.int)
  reals_zr = np.empty((state.n_rels, MAX_FACTS))
  active_array = state.rels.active.values
  for zr in range(n_rels):
    if not active_array[zr]:
      continue
    for zs in range(n_subs):
      for i in range(ref_counter[zr, zs]):
        refs_zr[zr, entity_valued_relation_counter[zr]] = entity_values[zr, zs, i]
        entity_valued_relation_counter[zr] += 1
      for i in range(real_counter[zr, zs]):
        reals_zr[zr, numeric_valued_relation_counter[zr]] = (
            numeric_values[zr, zs, i])
        numeric_valued_relation_counter[zr] += 1
  return dict(counter=np.asarray(entity_valued_relation_counter),
              refs=np.asarray(refs_zr),
              reals=np.asarray(reals_zr),
              real_counter=np.asarray(numeric_valued_relation_counter))


def ExtractFactsForEachRelation(state):
  """
  For each zr, return the facts (zo objects) corresponding to that zr
  """
  cdef long[:] numeric_fact_counter, active_array
  cdef long n_rels, n_subs, fact_index, count, zr, zs
  cdef double[:] numeric_facts
  cdef double[:, :] numeric_facts_by_relation
  cdef long[:, :] fact_index_array
  cdef double numeric_fact
  n_rels = state.n_rels
  n_subs = state.n_subs
  fact_index_array = state.facts_index
  numeric_facts = state.facts.real.values
  active_array = state.rels.active.values
  numeric_fact_counter = np.zeros(n_rels, np.int)
  numeric_facts_by_relation = np.zeros((n_rels, MAX_FACTS))
  for zr in range(n_rels):
    if not active_array[zr]:
      continue
    for zs in range(n_subs):
      fact_index = fact_index_array[zr, zs]
      numeric_fact = numeric_facts[fact_index]
      count = numeric_fact_counter[zr]
      numeric_facts_by_relation[zr, count] = numeric_fact
      numeric_fact_counter[zr] += 1
  return dict(counter=np.asarray(numeric_fact_counter), facts=np.asarray(numeric_facts_by_relation))


def TabulateTopicSubjectCounts(state):
  cdef long[:, :] counts
  cdef long[:] zt_array, table_array, zs_array
  cdef long n, N, zt
  N = len(state.rows)
  counts = np.zeros((state.n_topics, state.n_subs), np.int)
  zt_array = state.tables.zt.values
  table_array = state.rows.table.values
  zs_array = state.rows.zs.values
  for n in range(N):
    zt = zt_array[table_array[n]]
    counts[zt, zs_array[n]] += 1
  return np.asarray(counts)


def TabulateRangeSubjectCounts(state):
  cdef long[:, :] counts
  cdef long n, N
  cdef long[:] domain, entity_value
  counts = np.zeros((state.n_types, state.n_subs), np.int)
  cells = JoinCellsWithFactsTable(state)
  domain = cells['domain']
  entity_value = cells['ref']
  N = len(cells)
  for n in range(N):
    counts[domain[n], entity_value[n]] += 1
  return np.asarray(counts)


def TabulateTopicRelationCounts(state):
  cdef long[:, :] counts
  cdef long[:] zt_array, table_array, zr_array
  cdef long n, N, zt
  N = len(state.cols)
  counts = np.zeros((state.n_topics, state.n_rels), np.int)
  zt_array = state.tables.zt.values
  table_array = state.cols.table.values
  zr_array = state.cols.zr.values
  for n in range(N):
    zt = zt_array[table_array[n]]
    counts[zt, zr_array[n]] += 1
  return np.asarray(counts)


def TabulateSubjectTokenCounts(state, include_cells=True, exclude_table=-1, exclude_column=-1):
  cdef long[:, :] counts
  cdef long[:] entity_values, xo_ref, cell_types, zs, xs, column_ids, table_facts
  cdef long n, N, cell_type, zo, xo
  counts = np.zeros((state.n_subs, state.n_tokens), np.int)
  facts = JoinCellsWithFactsTable(state)
  entity_values = facts['ref']
  cell_types = facts['type']
  column_ids = state.cells.col.values
  table_facts = state.cells.table.values
  N = len(state.cells)
  xo_ref =state.cells.xo_ref
  zs = state.rows.zs.values
  xs = state.rows['xs'].values
  if include_cells:
    for n in range(N):
      column = column_ids[n]
      table = table_facts[n]
      if table==exclude_table and column==exclude_column:
        continue
      cell_type = cell_types[n]
      if cell_type != 0:
        continue
      xo = xo_ref[n]
      zo = entity_values[n]
      counts[zo, xo] += 1
  N = len(state.rows)
  for n in range(N):
    counts[zs[n], xs[n]] += 1
  return np.asarray(counts)


def TabulateRelationTokenCounts(state):
  cdef long[:, :] counts
  cdef long[:] zr, xr
  cdef long n, N
  counts = np.zeros((state.n_rels, state.n_tokens), np.int)
  zr = state.cols.zr.values
  xr = state.cols.xr.values
  N = len(state.cols)
  for n in range(N):
    counts[zr[n], xr[n]] += 1
  return np.asarray(counts)


def TabulateTopicCounts(state):
  return np.bincount(state.tables.zt, minlength=state.n_topics)


def TabulateRangeCounts(state):
  return np.bincount(state.rels.range_ref, minlength=state.n_types)

