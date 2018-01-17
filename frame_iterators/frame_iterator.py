
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class FrameIterator(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    """Initializes the FrameIterator."""
    pass

  @abc.abstractmethod
  def frame_iterator(self, filename, every_ms=1000, max_num_frames=-1):
    """Iterates over all frames of filename at a given frequency.

    Args:
      filename: Path to video file (e.g. mp4) or image directory.
      every_ms: The duration (in milliseconds) to skip between frames.
      max_num_frames: Maximum number of frames to process, taken from the
        beginning of the video.

    Yields:
      RGB frame with shape (image_height, image_width, channels)
    """
    pass
