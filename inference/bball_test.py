import mcmc
import data

def load_data():
  global s_full
  s_full = data.LoadState('data')

test_tables = [1872, 300]
s_init = data.ResetLatents(data.LoadState('test'))
trace=mcmc.Mcmc(s_init, 500)
s=trace.Last()
s.Plot()
