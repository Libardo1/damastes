"""Tools for tokenizing a corpus of strings."""

import re

import numpy as np
import pandas

import util

_ALIASES = {'sn par shady': ('sun', 'to', 'partial', 'shade')}

_COLORS = ['blue', 'orange', 'pink', 'gold', 'rose', 'white',
           'purple', 'lavender', 'yellow', 'red']


class Tokenizer(object):
  """Assigns a unique token to each string in a corpus.

  Attributes:
    token_to_str: A dictionary mapping integer tokens to normalized strings
    str_to_token: A dictionary mapping normalized strings to token IDs
  """

  def __init__(self):
    self.token_to_str = {}
    self.str_to_token = {}
    self.counter = 0

  def Normalize(self, s):
    black_words = ['spp']
    parts = re.split(r'\W+', s.strip())
    parts = [part.lower() for part in parts]
    parts = [part for part in parts if part not in black_words]
    is_color = np.asarray([part in _COLORS for part in parts])
    if np.any(is_color):
      parts = tuple(parts[np.flatnonzero(is_color)[0]])
    return tuple(parts)

  def TokenForStr(self, s, similarity_thres=None):
    """Returns a token ID for a given string.

    If the string has already been assigned a token, return that token.
    Otherwise, create a new token. The string is normalized before
    being checked if it's already been assigned a token.

    Args:
      s: A (possibly-unnormalized) string.
      similarity_thres: A number from 0 to 1 representing how similar two
      strings have to be to warrant being given the same token.

    Returns:
      An integer token ID.
    """
    blacklist = ['perennial', 'bergenia']
    aliases = _ALIASES
    s_norm = self.Normalize(s)
    s_norm_joined = ' '.join(s_norm)
    if s_norm_joined in aliases:
      s_norm = aliases[s_norm_joined]
    if s_norm not in self.str_to_token:
      if similarity_thres is not None:
        for token in self.token_to_str:
          s_token = self.StrForToken(token)
          sim = util.StringSim(s_token, s_norm_joined)
          if sim > similarity_thres and (s_norm_joined not in blacklist):
            self.str_to_token[s_norm] = token
            return token
      self.str_to_token[s_norm] = self.counter
      self.counter += 1
    token = self.str_to_token[s_norm]
    self.token_to_str[token] = s_norm
    return token

  def StrForToken(self, token):
    if token in self.token_to_str:
      s = ' '.join(self.token_to_str[token])
    else:
      s = 'Token %d' % token
    return s

  def StringList(self, n_tokens=None):
    if n_tokens is None:
      n_tokens = self.TokenCount()
    strings = [self.StrForToken(token) for token in range(n_tokens)]
    strings = pandas.Series(strings, name='str')
    return strings

  def TokenCount(self):
    return len(self.token_to_str)
