"""Performs end-to-end inference.
"""

import logging

import util, data, mcmc, prior_sampler


if __name__=='__main__':
  util.ConfigureLogging()
  logging.info('Performing end-to-end inference')
  flags = util.GetFlags(use_command_line=True)
  initial_state = data.LoadStateFromFile(flags['dataset'])
  prior_sampler.ResampleLatents(initial_state)
  trace = mcmc.Mcmc(initial_state)
  trace.Save(flags['trace_id'])
  logging.info('Inference complete')

