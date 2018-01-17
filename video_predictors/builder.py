
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from video_predictors.advise_video_predictor import ADVISEVideoPredictor
from video_predictors.youtube8m_video_predictor import Youtube8MVideoPredictor

def build(name):
  """Builds frame iterator based on the given name.

  Args:
    name: the name of the specific frame iterator.

  Returns:
    An instance of FrameIterator.

  Raises:
    ValueError: if name is invalid.
  """

  if 'ADVISEVideoPredictor' == name:
    return ADVISEVideoPredictor()

  if 'Youtube8MVideoPredictor' == name:
    return Youtube8MVideoPredictor()

  raise ValueError('Invalid name %s.' % name)
