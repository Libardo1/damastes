#!/usr/bin/env python
import argparse
import subprocess


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('mode', help='what build action to perform')

  args = parser.parse_args()
  if args.mode == 'build':
    print 'running build'
    subprocess.call(['python', 'setup.py', 'build_ext', '--inplace'])
    print 'build completed'
  elif args.mode == 'clean':
    print 'cleaning'
    subprocess.call('rm -rf build', shell=True)
    subprocess.call('rm -rf *.so', shell=True)
    subprocess.call('rm -rf *.cpp', shell=True)
    subprocess.call('rm -f *.pyc', shell=True)
    print 'done cleaning'
  else:
    print 'Command %r not recognized' % args.mode
