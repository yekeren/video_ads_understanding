
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from frame_iterators.frame_iterator import FrameIterator

CAP_PROP_POS_MSEC = 0


class VideoCaptureFrameIterator(FrameIterator):
  def __init__(self):
    """Initializes the FrameIterator."""
    pass

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
    video_capture = cv2.VideoCapture()
    if not video_capture.open(filename):
      print >> sys.stderr, 'Error: Cannot open video file ' + filename
      return

    last_ts = -99999  # The timestamp of last retrieved frame.
    num_retrieved = 0

    while max_num_frames < 0 or num_retrieved < max_num_frames:
      # Skip frames
      while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
        if not video_capture.read()[0]:
          return

      last_ts = video_capture.get(CAP_PROP_POS_MSEC)
      has_frames, frame = video_capture.read()
      if not has_frames:
        break
      yield frame[:, :, ::-1]
      num_retrieved += 1
