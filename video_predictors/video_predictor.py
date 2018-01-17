
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class VideoPredictor(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    """Initializes the VideoPredictor."""
    pass

  @abc.abstractmethod
  def predict(self, rgb_images):
    """Give a bunch of video frames, predict the info of the video clip.

    Args:
      rgb_images: A list of (image_height, image_width, channes) image frames.
    """
    pass
