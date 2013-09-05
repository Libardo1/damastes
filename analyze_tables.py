import cPickle
import subprocess

from google3.pyglib import app
from google3.pyglib import flags
from google3.pyglib import gfile
from google3.pyglib import logging

from google3.knowledge.graph.vault.open_world.inference import data, mcmc, prior_sampler
from google3.knowledge.graph.vault.open_world.webtables import importer



FLAGS = flags.FLAGS
flags.DEFINE_boolean('force_import', False, 'Force reloading of tables, even if previously imported')
flags.DEFINE_string('dataset_name', 'botany', 'The name of the dataset to analyze')
flags.DEFINE_string('dataset_directory', '/cns/gd-d/home/malmaud/datasets', 'Directory containing the cached datasets')
flags.DEFINE_integer('n_iterations', 100, 'The number of sweeps of Gibbs inference to run')
flags.DEFINE_integer('thinning_interval', 10, 'The thinning interval of the trace of the MCMC chain')
flags.DEFINE_string('output_file', '/cns/gd-d/home/malmaud/trace.pickle', 'File location to store the MCMC trace')


def LoadTablesFromCache(dataset_name):
  dataset_directory = FLAGS.dataset_directory
  try:
    tables = data.LoadStateFromFile(dataset_name, dataset_directory)
    return tables
  except IOError:
    return None


def main(unused_args):
  logging.info('Beginning analysis')
  logging.fatal('Not currently operation')
  dataset_name = FLAGS.dataset_name
  dataset_directory = FLAGS.dataset_directory
  force_import = FLAGS.force_import
  n_iterations = FLAGS.n_iterations
  n_thinning = FLAGS.thinning_interval
  processed_tables = LoadTablesFromCache(dataset_name)
  if (not processed_tables) or force_import:
    logging.info('Importing tables from Webtables')
    raw_tables = importer.ImportWebTables(output_to_file=False)
    processed_tables = data.InitializeStateFromData(raw_tables)
    data.SaveState(processed_tables, dataset_name, dataset_directory)
  trace = mcmc.Mcmc(processed_tables, n_iters=n_iterations, skip=n_thinning)
  output_file_name = FLAGS.output_file
  with gfile.GFile(output_file_name, 'w') as output_file:
    cPickle.dump(trace, output_file, protocol=2)
  logging.info('Analysis finished')

if __name__ == '__main__':
  app.run()