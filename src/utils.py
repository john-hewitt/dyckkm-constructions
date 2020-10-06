"""
Utilities for determining paths to corpora, results, models
given config dictionaries describing an experiment, as well
as determining canonical vocabulary ordering
"""

import os
import string
import re
import copy

SCALE_RECURRENT=1e4
SCALE_DISTRIBUTION=1e4

def get_identifier_iterator():
  ids = iter(list(string.ascii_lowercase))
  k = 1
  while True:
    try:
      str_id = next(ids)
    except StopIteration:
      ids = iter(list(string.ascii_lowercase))
      k += 1
      str_id = next(ids)
    yield str_id*k

def get_vocab_of_bracket_types(bracket_types):
  id_iterator = get_identifier_iterator()
  ids = [next(id_iterator) for x in range(bracket_types)]
  vocab = {x: c for c, x in enumerate(['(' + id_str for id_str in ids] + [id_str + ')' for id_str in ids] + ['END'])}
  return vocab, ids
