"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()
    #return list(f.read().replace("\n", "<eos>"))


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  #counter = {k:v for k, v in counter.items() if v >= 10}
  #repl = []
  #for k, v in counter.items():
  #  if v >= 6:
  #    repl.append(k)
  #    #data[:] = [x if x == k else '<unk>' for x in data]
  #print(repl)

  #print(len(data))
  #print(len(repl))
  
  #for i in range(len(data)):
  #  if data[i] in repl:
  #    data[i] = '<unk>'

  #with open('text8_6.txt', 'a') as f:
  #  for wrd in data:
  #    f.write(wrd+" ")
  #counter['<unk>'] = 1
  #counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  id_to_word = dict((v, k) for k, v in word_to_id.items())

  return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def wiki_raw_data(data_path=None, prefix="ptb"):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, 'train2')
  valid_path = os.path.join(data_path, "valid2")
  test_path = os.path.join(data_path, "valid2")

  word_to_id, id_2_word = _build_vocab(train_path)

  vocabulary = len(word_to_id)

  #data = _file_to_word_ids(train_path, word_to_id)
  #data = _file_to_word_ids(train_path, word_to_id)
  #data = _file_to_word_ids(train_path, word_to_id)
  #length = len(data)
  """c1 = length*98//100
  c2 = length*99//100
  train_data = data[0:c1]
  valid_data = data[c1:c2]
  test_data = data[c2:]"""
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  return train_data, valid_data, test_data, word_to_id, id_2_word


def wiki_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.
  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)
