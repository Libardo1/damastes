"""Defines the State object and methods to load a state from data.
"""
import copy
import os

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas

import cPosteriors
import tokenizer
import util

SIZE = 50
N_RELS = SIZE  # Maximum supported number of relations
N_SUBS = SIZE  # Maximum number of subjects
N_TOPICS = 1  # Maximum amount of topics
N_RANGES = SIZE  # Maximum amount of relation ranges
N_TOKENS = 300  # Maximum amount of unique string tokens


class State(object):
  """Represents the state of the MCMC chain at a given moment of time.

  Stores observed and latent data as Pandas DataFrames.

  Attributes:
    entity_tokenizer: Tokenizer object that maps between tokens
      and entity strings
    col_tokenizer: Tokenizer object that maps between tokens and column header
      strings.
    raw_tables: A list of dictionaries giving the raw string information for
      the stored tables, before preprocessing
    n_subs: Maximum number of subjects
    n_rels: Maximum number of relations
    n_topics: Maximum number of topics
    n_tokens: Maximum number of tokens
    n_types: Maximum number of relation types (ie distinct ranges)
    data_dims: An array storing the dimensions of each table.
      data_dims[table, 0] and data_dims[table, 1] are the number of rows and
      columns in table 'table' respectively.
    The following are Pandas DataFrames:
    rels: One record per latent relation
    subs: One record per latent subject
    cols: One record per column in each table
    rows: One record per row in each table
    tables: One record per table
    facts: One record for each combination of subject and relation
    hypers: A Pandas Series (ie a vector that can be indexed with a string key)
     storing the model hyperparameters
    accept_log: A boolean array storing a history of accepted MH moves

  See https://docs.google.com/a/google.com/document/d/1MJTTWiMSWzVuLph7-6usaYD0otW-GRIaqT5QBYwuAak/edit for a detailed description of the state parameterization.
  """
  
  def __init__(self):
    self.entity_tokenizer = tokenizer.Tokenizer()
    self.col_tokenizer = tokenizer.Tokenizer()
    self.raw_tables = []  # TODO(malmaud): Don't waste space by storing this in the state

  def Copy(self):
    return copy.deepcopy(self)  # todo don't copy data unless it's a geweke test

  def SetSize(self, data_dims):
    """Allocate the dataframes for tables of specified dimensions.

    Args:
      data_dims: An array with one entry per table, where each entry is a tuple
        with the number of rows and columns the table.
    """
    self.n_subs = N_SUBS
    self.n_rels = N_RELS
    self.n_topics = N_TOPICS
    self.n_tokens = N_TOKENS
    self.n_tables = len(data_dims)
    self.n_types = N_RANGES
    self.data_dims = np.array(data_dims, int)
    self.rels = pandas.DataFrame(dict(range_real_mu=np.empty(self.n_rels),
                                      range_ref=np.empty(self.n_rels, int),
                                      range_real_sd=np.empty(self.n_rels),
                                      active=np.repeat(1, self.n_rels)))
    self.rels.index.names = ['zr']
    self.subs = pandas.DataFrame(dict(active=np.repeat(1, self.n_subs)))
    self.subs.index.names = ['zs']
    d_cols = []
    d_rows = []
    d_cells = []
    d_table = []
    for t, (rows, cols) in enumerate(data_dims):
      d_table.append(dict(zt=0, topic_str='', url=''))
      for col in range(cols):
        d_cols.append(dict(table=t, col=col, zr=0, xr=0, topic=0, type=0))
      for row in range(rows):
        d_rows.append(dict(table=t, row=row, zs=0, xs=0, topic=0))
      for col in range(cols):
        for row in range(rows):
          d_cells.append(dict(table=t, row=row, col=col,
                              xo_ref=0, xo_real=0.))
    self.cols = pandas.DataFrame(d_cols)
    self.rows = pandas.DataFrame(d_rows)
    self.cells = pandas.DataFrame(d_cells)
    self.cells.set_index(['table', 'row', 'col'], inplace=True)
    self.cols.set_index(['table', 'col'], inplace=True)
    self.rows.set_index(['table', 'row'], inplace=True)
    self.tables = pandas.DataFrame(d_table)
    self.tables.index.names = ['table']
    d_facts = []
    for sub in range(self.n_subs):
      for rel in range(self.n_rels):
        d_facts.append(dict(zs=sub, zr=rel, real=0., ref=0))
    self.facts = pandas.DataFrame(d_facts)
    self.facts.set_index(['zr', 'zs'], inplace=True)
    self.hypers = pandas.Series()
    self.accept_log = []
    self.sub_token_weights = np.empty((self.n_subs, self.n_tokens), np.int)
    self.rel_token_weights = np.empty((self.n_rels, self.n_tokens), np.int)
    self.ConstructIndices()

  def ConvertToThirdNormalForm(self):
    """Convert the database represented by this state into Third Normal Form.

    In Third Normal Form (3NF), the value each column of a dataframe is a
    function of (and only of) the primary key of that dataframe.
    See http://en.wikipedia.org/wiki/Third_normal_form.

    Also gives each table canonical unique indices.
    """
    self.cells = (self.cells.reset_index()[['table', 'row', 'col',
                                            'xo_ref', 'xo_real']].
                  set_index(['table', 'row', 'col'], drop=False).
                  sortlevel('table'))
    self.rows = (self.rows.reset_index()[['table', 'row', 'zs', 'xs']].
                 set_index(['table', 'row'], drop=False).
                 sortlevel('table'))
    self.cols = (self.cols.reset_index(drop=True)[['table', 'col', 'xr',
                                                   'zr', 'type']].
                 set_index(['table', 'col'], drop=False).
                 sortlevel('table'))
    self.facts = (self.facts.reset_index()[['zr', 'zs', 'real', 'ref']].
                  set_index(['zr', 'zs'], drop=False).
                  sortlevel('zr'))
    self.tables = self.tables.sort_index()
    self.ConstructIndices()

  def ConstructIndices(self):
    """Computes scalar indices for accessing the dataframes.

    Primarily of benefit to Cython code which bypasses the Pandas indexing
    mechanism for performance reasons.
    """
    s = self
    s.facts_index = util.ConstructNPIndex(s.facts.index)
    s.cells_index = util.ConstructNPIndex(s.cells.index)
    s.cols_index = util.ConstructNPIndex(s.cols.index)
    s.rows_index = util.ConstructNPIndex(s.rows.index)

  def GetCellsJoinedWithFactsTable(self):
    """Table cells joined with their latent properties.

    Returns:
      A dataframe of cells that have been augmented with the cell's latent
      relation, subject, and factual referent
    """
    c = self.cells  # TODO(malmaud): implement caching for the joining functions
    c = c.join(self.rows, on=['table', 'row'], rsuffix='_rows')
    c = c.join(self.cols, on=['table', 'col'], rsuffix='_cols')
    c = c.join(self.facts, on=['zr', 'zs'], rsuffix='_facts')
    c = c[['table', 'row', 'col', 'xo_ref',
           'xo_real', 'zs', 'zr', 'real', 'ref', 'type']]
    return c

  def GetFactsWithRanges(self):
    """Table of facts joined with their range parameters.

    Returns:
      A dataframe of facts augmented with the parameters of the range of the
      fact's relation.
    """
    f = self.facts.join(self.rels, on='zr')
    return f

  def VisualizeWeights(self,
                       weight_matrix,
                       is_log=False,
                       xlabel='x',
                       ylabel='y',
                       file_name='weights',
                       output_dir='output'):
    plt.figure()
    util.VisualizeWeights(weight_matrix, is_log=is_log)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(output_dir, "%s.svg" % file_name))

  def ProduceFigures(self, output_dir='output'):
    self.VisualizeWeights(self.sub_token_weights,
                          False,
                          'Token',
                          'Subject',
                          'sub_token_weights',
                          output_dir)
    self.VisualizeWeights(self.rel_token_weights,
                          False,
                          'Token',
                          'Relation',
                          'rel_token_weights',
                          output_dir)
    self.VisualizeWeights(cPosteriors.RowPosterior(self)[0],
                          True,
                          'Subject',
                          'Row',
                          'row_posterior',
                          output_dir)
    self.VisualizeWeights(cPosteriors.UpdateColumnPosterior(self),
                          True,
                          'Relation',
                          'Column',
                          'col_posterior',
                          output_dir)
    self.VisualizeWeights(self.topic_sub_weights,
                          False,
                          'Subject',
                          'Topic',
                          'topic_sub_weights',
                          output_dir)

  def Plot(self, output_format='html',
           plot_id=None,
           produce_figs=False,
           output_dir='output'):
    """Produce a visualization of the state.

    Args:
      output_format: What format to visualize the state in.
        Only 'html' is currently supported.
        In html mode, a file called 'state.html' is created in the
        output directory.
      plot_id: Optionally, an integer appended to the
        filename of the HTML output.
      produce_figs: A boolean on whether to generate figures to include
        in the HTML file. Can take a lot of time to produce.
      output_dir: Directory for storing the output.
    """
    if plot_id is None:
      filename = 'state.html'
    else:
      filename = 'state_%d.html' % plot_id
    filename = os.path.join(output_dir, filename)
    if output_format not in ['html']:
      raise util.TablesException(
          'Format %r not recognized' % output_format)
    if output_format == 'html':
      if produce_figs:
        self.ProduceFigures(output_dir)
      template = jinja2.Template(open('templates/state_template.jinja2').read())
      html = template.render(s=self)
      with open(filename, 'w') as f:
        f.write(html)

  def GetFactMatrix(self):
    """Returns a matrix X such that X[i,j] = Object of subject i, relation j.
    """
    return self.facts.pivot(index='zr', columns='zs')

  def InsertTokens(self):
    """Augment cells with the strings their tokens refer to."""
    entity_tokens = self.entity_tokenizer.StringList()
    rel_tokens = self.col_tokenizer.StringList()
    self.rows = self.rows.join(entity_tokens, on='xs')
    self.cols = self.cols.join(rel_tokens, on='xr')
    self.cells = self.cells.join(entity_tokens, on='xo_ref')

  def ComputeTokenDistribution(self):
    """Compute a histogram to string usage.

    Returns:
      A Series of usage counts, indexed by token ID. Sorted descending.
    """
    cells = self.GetCellsJoinedWithFactsTable()
    token_count = cells[cells.type == 0].groupby('xo_ref').size().reset_index()
    strings = self.entity_tokenizer.StringList()
    d = token_count.join(strings, on='xo_ref')
    d.columns = pandas.Index(['xo', 'count', 'token'])
    d.sort('count', ascending=False, inplace=True)
    d.set_index('xo', inplace=True)
    return d

  def FindTableWithToken(self, observed_token):
    """Finds an example of a table expressing a certain token.

    Args:
      observed_token: An integer token ID

    Returns:
      A table in raw form and a table URL, if a table is found.
      Otherwise, returns None.
    """
    # First check the cells
    f = self.cells[self.cells.xo_ref == observed_token]
    if not f:
      # Then try checking the row subjects
      f = self.rows[self.rows['xs'] == observed_token]
    if not f:
      return None
    table = f.table.iloc[0]
    url = self.tables.loc[table, 'url']
    return self.raw_tables[table], url

