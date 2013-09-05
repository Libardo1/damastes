"""Tools for importing, saving, and loading data.
"""
import cPickle
import locale
import logging
import os
import re
import subprocess

import dateutil
import numpy as np
import pandas

import prior_sampler
import state
import util

from google3.pyglib import gfile


def GetDataDir(base, part, dataset_directory='input'):
  return os.path.join(dataset_directory, base, '%s.pickle' % part)


def LoadStateFromFile(dataset_name, directory='input'):
  file_name = GetDataDir(dataset_name, 'state', directory)
  with gfile.GFile(file_name) as input_file:
    state_for_dataset = cPickle.load(input_file, 2)
  return state_for_dataset


def SaveState(state_to_save, dataset_name, directory):
  file_path = os.path.join(directory, dataset_name)
  with gfile.GFile(file_path, 'w') as output_file:
    cPickle.dump(state_to_save, output_file, 2)


def GetSubsetOfTables(full_dataset, table_ids):
  tables = [full_dataset.raw_tables[table_id] for table_id in table_ids]
  desc = ExtractObsDimensionsFromRawTables(tables)
  full_dataset = prior_sampler.SamplePrior(desc)
  full_dataset.InitializeFromData(tables)
  return full_dataset


def ResetLatentVariables(original_state):
  table_ids = range(len(original_state.raw_tables))
  return GetSubsetOfTables(original_state, table_ids)


def LoadBotanyDataset(max_rows=10, max_cols=4,
                      max_tables=5, pull=False, matching=True):
  if pull:
    subprocess.call('fileutil cp /cns/gd-d/home/malmaud/tables.pickle .',
                    shell=True)
  with open('output/botany_tables.pickle') as f:
    tables = cPickle.load(f)

  # The following block applies hacky alterations to data
  # for demonstration purposes
  t = tables[3]
  t['xr'] = t['xo'][1]
  t['xo'] = t['xo'][2:53]
  t['xs'] = t['xs'][2:53]
  t['xr'] = t['xr'][1:]
  t['xo'] = [row[1:] for row in t['xo']]
  t = tables[0]
  t['xo'] = t['xo'][1:]
  t['xs'] = t['xs'][1:]
  if matching:
    for t in tables:
      t['xs'] = np.asarray(t['xs'])
      t['xr'] = np.asarray(t['xr'])
      t['xo'] = np.asarray(t['xo'])
    rows =[
        [4, 8, 10, 14, 16, 19, 33, 35, 42, 46, 50, 57, 70, 100, 106],
        [1, 3, 4, 6, 10, 22, 29, 33, 47, 53, 73],
        [0, 2, 8, 9, 10, 11, 12, 13, 15, 17, 20, 22, 27],
        [0, 8, 12, 15, 22],
    ]
    for t, n in zip(tables, rows):
      t['xs'] = t['xs'][n]
      t['xo'] = t['xo'][n, :]
    t = tables[0]
    t['xr'][2] = 'sun level'
    xr = ['Blooming months', 'Pollen color']
    xs = ['Aster', 'Marigold', 'Fireweed']
    xo = [
        ['Sep', 'red'],
        ['Jun', 'orange'],
        ['Jul', 'blue']]
    tables = tables[:4]
    tables.append(dict(xr=xr, xs=xs, xo=xo,
                       table_id='http://en.wikipedia.org/wiki/Pollen_source',
                       x_column='Scientific name'))
    t = tables[2]
    t['xr'] = ['Timespan']
    cells = t['xo'][:, 1]
    cells[cells == 'Annual'] = 'yearly'
    t['xo'] = t['xo'][:, 0:1]
    t['xo'][:, 0] = cells

  else:
    for t in tables:
      t['xo'] = t['xo'][:max_rows]
      t['xs'] = t['xs'][:max_rows]
    for t in tables:
      t['xo'] = [row[:max_cols] for row in t['xo']]
      t['xr'] = t['xr'][:max_cols]

  # End hacky block
  tables = tables[:max_tables]
  botany_tables = InitializeStateFromData(tables, prior_sampler.SamplePrior)
  zs = [0, 1, 2, 3, 4, 5, 4, 6, 7, 8, 4]
  botany_tables.rows.loc[range(len(zs)), 'zs'] = zs
  SaveState(botany_tables, 'botany')
  botany_tables.Plot()
  return botany_tables


def ExtractObsDimensionsFromRawTables(tables):
  """Calculate the dimensionality of a set of tables.

  Args:
    tables: A set of tables in dictionary-of-strings format.

  Returns:
    A 2D array of table dimensions, with one row per table. The first column
    is the number of rows in each table; the second the number of columns.
  """
  dimensions = []
  for table in tables:
    cells = table['xo']
    rows = len(cells)
    cols = len(cells[0])
    dimensions.append((rows, cols))
  return dimensions

# Cells with these symbols are considered to be blank
MISSING_SYM = ['', '*', '-', '.', 'na', 'NA', 'n/a', 'N/A']

# Various regular expressions used for parsing cells in tables
HEIGHT_RE_1 = re.compile(r'(\d+)\'?\s*-\s*(\d+)"?')
HEIGHT_RE_2 = re.compile(r'(\d+)\'\s*(\d+)"')
TIME_RE = re.compile(r'(\d+)\s*:\s*(\d+)')
PERCENT_RE = re.compile(r'([\d\.]+)\s*%')
DOLLAR_RE = re.compile(r'\$([\d\.,]+).*')
DATE_RE = re.compile(r'(\d+)\s*/\s*(\d+)\s*/\s*(\d+)')
HEIGHT_RANGE_RE = re.compile(r'(\d+)\s*(to|-)\s*(\d+)(.*)')

# Categories of data in table cells. Each cell is determined to be one of these:
REF = 0  # A string which refers to an entity.
REAL = 1  # A real-valued numeric literal.
BLANK = 2  # A cell whose value is missing.
PROSE = 3  # A cell with a string of more than 3 words,
           # which is assumed to be prose which doesn't refer to an entity.


def PreprocessColumn(cells):
  """Returns a canonical representation of the cells in a column.

  Applies heuristics to determine what category of data the majority of cells in a column are   referring to.
  For numeric data, the unit of the column is also heuristically determined.
  Based on that analysis, converts the cells into a canonical representation of
  instances of that data category.

  For example, cells which are determined to refer to heights are
  converted to a real value in inches, putting all heights on the same scale.

  Money, times, and dates are also converted to a canonical form.

  Args:
    cells: A list of strings appearing in a given column.

  Returns:
    A tuple with three elements.
    The first element is the canonical representation of the cells.
    The second element is a binary vector indicating which
    cells have the modal category of this column,
    as some columns might have mixed types of cells.
    The third element is the ID of the modal category of the cells in the column,
    which is taken to be the category of the column as a whole.
  """
  proc_cells = pandas.pandas.DataFrame([DecideCellType(cell) for cell in cells],
                                       columns=['type', 'value'])
  missing = np.flatnonzero(proc_cells.type == BLANK)
  proc_cells['value'][missing] = proc_cells['value'][missing - 1]
  type_counts = proc_cells.groupby('type').size()
  type_counts.sort(ascending=False)
  mode_type = type_counts.index[0]
  return proc_cells['value'], proc_cells['type'] == mode_type, mode_type


def DecideCellType(cell, extensive_checking=False):
  s = cell.strip()
  if extensive_checking:
    if s in MISSING_SYM:
      return BLANK, ''
    split = s.split()
    if len(split) > 2:
      return PROSE, s
    try:
      return REAL, locale.atof(split[0])
    except ValueError:
      m = HEIGHT_RE_1.match(s)
      if not m:
        m = HEIGHT_RE_2.match(s)
      if m:
        feet = locale.atof(m.group(1))
        inches = locale.atof(m.group(2))
        return REAL, feet + inches/12.
      m = TIME_RE.match(s)
      if m:
        minute = locale.atof(m.group(1))
        second = locale.atof(m.group(2))
        return REAL, minute + second/60.
      m = PERCENT_RE.match(s)
      if m:
        return REAL, locale.atof(m.group(1))
      m = DOLLAR_RE.match(s)
      if m:
        return REAL, locale.atof(m.group(1))
      try:
        date = dateutil.parser.parse(s)
        return REF, date.date().ctime()
      except ValueError:
        return REF, s
      except TypeError:
        return REF, s
  else:
    m = HEIGHT_RANGE_RE.match(s)
    if m:
      height1 = float(m.group(1))
      height2 = float(m.group(3))
      symbol = m.group(4)
      mu = np.mean([height1, height2])
      if symbol in ["'"]:
        mu *= 12  # feet to inches
      return REAL, mu
    return REF, s


def IsGoodTable(table):
  if not table['xo']:
    return False
  if not table['xo'][0]:
    return False
  return True


def PreprocessTable(table):
  if not IsGoodTable(table):
    return None
  new_table = {}
  xo = np.array(table['xo'])
  xr = np.array(table['xr'])
  n_cols = xo.shape[1]
  n_rows = xo.shape[0]
  new_table['type'] = []
  good_cols = np.repeat(False, n_cols)
  good_rows = np.repeat(True, n_rows)
  for col in range(n_cols):
    cells, col_good_rows, col_type = PreprocessColumn(xo[:, col])
    good_rows &= col_good_rows
    xo[:, col] = cells
    if col_type in [REF, REAL]:
      new_table['type'].append(col_type)
      good_cols[col] = True
  if sum(good_cols) == 0 or sum(good_rows) == 0:
    return None
  xo_filtered = xo[good_rows, :]
  xo_filtered = xo_filtered[:, good_cols]
  new_table['xo'] = xo_filtered
  new_table['xr'] = xr[good_cols]
  new_table['xs'] = np.array(table['xs'])[good_rows]
  new_table['table_id'] = table['table_id']
  new_table['topic_str'] = table['x_column']
  return new_table


def InitializeStateFromData(tables, initializer=None):
  """Create a state given a set of raw tables.

  Args:
    tables: A list of preprocessed tables in dictionary-of-strings format.
     Each table is a dictionary d where d['xs'] is a list of strings of the
     subject column, d['xr'] is a list of strings of the column headers,
     d['xo'] is an array such that ['xo'][row, col] is the string at
     the corresponding part in the table,  d['table_id'] is the Google
     ID of the table (as used by Webtables).
    initializer: Function which returns the initial state
      of the latent variables. By default, draws from the prior.
  """
  locale.setlocale(locale.LC_ALL, 'en_US')
  tables = [PreprocessTable(table) for table in tables]
  tables = [table for table in tables if table is not None]
  table_dimensions = ExtractObsDimensionsFromRawTables(tables)
  if initializer is None:
    state_for_tables = state.State()
    state_for_tables.SetSize(table_dimensions)
  else:
    state_for_tables = initializer(table_dimensions)
  table_id = -1
  for table_idx, table_raw in enumerate(tables):
    logging.info('Loading table %d', table_idx)
    table = table_raw
    if not table:
      continue
    table_id += 1
    state_for_tables.raw_tables.append(table_raw)
    state_for_tables.tables.loc[table_id, 'url'] = table['table_id']
    state_for_tables.tables.loc[table_id, 'topic_str'] = table['topic_str']
    logging.info('adding url %r from table %d',
                 table['table_id'], table_id)
    xo = table['xo']
    try:
      xs = table['xs']
    except KeyError:
      xs = ['Row %d' % i for i in range(len(xo))]
    try:
      xr = table['xr']
    except KeyError:
      xr = ['Col %d' % i for i in range(len(xo[0]))]
    types = table['type']
    for row_id, row in enumerate(xo):
      for col_id, cell in enumerate(row):
        if types[col_id] == 0:
          token = state_for_tables.entity_tokenizer.TokenForStr(cell)
          state_for_tables.cells.loc[(table_id, row_id, col_id), 'xo_ref'] = token
        elif types[col_id] == 1:
          state_for_tables.cells.loc[(table_id, row_id, col_id), 'xo_real'] = (
              locale.atof(cell))
        else:
          raise util.TablesException(
              'Datatype %r not recognized' % types[col_id])
    for col_id, col_type in enumerate(types):
      state_for_tables.cols.loc[(table_id, col_id), 'type'] = col_type
    for row_id, row in enumerate(xs):
      token = state_for_tables.entity_tokenizer.TokenForStr(row)
      state_for_tables.rows.loc[(table_id, row_id), 'xs'] = token
    for col_id, col in enumerate(xr):
      token = state_for_tables.col_tokenizer.TokenForStr(col)
      state_for_tables.cols.loc[(table_id, col_id), 'xr'] = token
  return state_for_tables



