
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class ClipIterator(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    """Initializes the FrameIterator."""
    pass

  @abc.abstractmethod
  def clip_iterator(self, rgb_images):
    """Iterates over all clips of a given video.

    Args:
      rgb_images: A list of (image_height, image_width, channes) image frames.

    Yields:
      Python dict representing video clip, involving following fields:
        position: the index of the start frame.
        length: the number of frames in the clip.
        rgb_images: rgb images in the clip.
    """
    pass
