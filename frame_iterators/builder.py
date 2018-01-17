
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from frame_iterators.video_capture_frame_iterator import VideoCaptureFrameIterator

def build(name):
  """Builds frame iterator based on the given name.

  Args:
    name: the name of the specific frame iterator.

  Returns:
    An instance of FrameIterator.

  Raises:
    ValueError: if name is invalid.
  """

  if 'VideoCaptureFrameIterator' == name:
    return VideoCaptureFrameIterator()

  raise ValueError('Invalid name %s.' % name)
