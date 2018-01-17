
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from clip_iterators.hist_correl_clip_iterator import HistCorrelClipIterator
from clip_iterators.dense_optical_flow_clip_iterator import DenseOpticalFlowClipIterator


def build(name):
  """Builds frame iterator based on the given name.

  Args:
    name: the name of the specific frame iterator.

  Returns:
    An instance of FrameIterator.

  Raises:
    ValueError: if name is invalid.
  """

  if 'HistCorrelClipIterator' == name:
    return HistCorrelClipIterator()

  if 'DenseOpticalFlowClipIterator' == name:
    return DenseOpticalFlowClipIterator()

  raise ValueError('Invalid name %s.' % name)
