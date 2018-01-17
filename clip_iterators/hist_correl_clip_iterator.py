
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from clip_iterators.clip_iterator import ClipIterator


def _image_to_hist(image, crop_ratio=0.12):
  # Process center crop.
  height, width, _ = image.shape
  h_border = int(height * crop_ratio)
  w_border = int(width * crop_ratio)
  image = image[h_border: height - h_border, w_border: width - w_border, :]

  # Compute hsv histogram.
  image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

  hist = cv2.calcHist([image], channels=[0, 1, 2], mask=None, 
      histSize=[8, 8, 8], ranges=[0, 180, 0, 256, 0, 256])
  cv2.normalize(hist, hist)

  return hist


class HistCorrelClipIterator(ClipIterator):
  def __init__(self):
    """Initializes the ClipIterator."""
    self._distance_threshold = 0.8

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

    def _distance(a, b):
      return 1 - cv2.compareHist(a, b, cv2.HISTCMP_CORREL)

    histograms = [_image_to_hist(rgb) for rgb in rgb_images]

    clips = [[0]]
    for frame_id in xrange(1, len(rgb_images)):
      clip = clips[-1]
      if _distance(histograms[clip[0]], 
          histograms[frame_id]) < self._distance_threshold:
        clip.append(frame_id)
      else:
        clips.append([frame_id])

    result = []
    for clip in clips:
      result.append({
          'position': clip[0],
          'length': len(clip),
          'rgb_images': [rgb_images[frame_id] for frame_id in clip],
          })
    return result
