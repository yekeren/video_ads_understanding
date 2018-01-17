
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from clip_iterators.clip_iterator import ClipIterator


def _crop(image, crop_ratio=0.12):
  # Process center crop.
  height, width, _ = image.shape
  h_border = int(height * crop_ratio)
  w_border = int(width * crop_ratio)
  return image[h_border: height - h_border, w_border: width - w_border, :]


def _gray_to_optical_flow(image_prev, image_next):
  #flow = cv2.calcOpticalFlowFarneback(
  #    image_prev, image_next, 
  #    flow=None, 
  #    pyr_scale=0.5, 
  #    levels=3, 
  #    winsize=15, 
  #    iterations=3, 
  #    poly_n=5, 
  #    poly_sigma=1.2, 
  #    flags=0)

  flow = cv2.calcOpticalFlowFarneback(
      image_prev, image_next, 
      flow=None, 
      pyr_scale=0.5, 
      levels=3, 
      winsize=1, 
      iterations=3, 
      poly_n=7, 
      poly_sigma=1.5, 
      flags=0)
  return flow


def _optical_flow_to_polar(flow):
  return cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])


class DenseOpticalFlowClipIterator(ClipIterator):
  def __init__(self):
    """Initializes the ClipIterator."""
    self._initial_magnitude_threshold = 1.0
    self._max_video_clips = 40

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
    # Use opencv to compute the optical flow images.
    gray_images = [cv2.cvtColor(_crop(rgb), cv2.COLOR_RGB2GRAY) for rgb in rgb_images]
    flow_images = [_gray_to_optical_flow(gray_images[i - 1], gray_images[i]
        ) for i in xrange(1, len(gray_images))]
    mag_and_ang = [_optical_flow_to_polar(flow) for flow in flow_images]

    mags = [mag.mean() for mag, _ in mag_and_ang]
    magnitude_threshold = self._initial_magnitude_threshold

    while (np.array(mags) > magnitude_threshold
        ).sum() > self._max_video_clips:
      magnitude_threshold += 1

    clips = [[0]]
    for index, mag in enumerate(mags):
      frame_id = index + 1
      clip = clips[-1]

      if mag >= magnitude_threshold:
        clips.append([frame_id])
      else:
        clip.append(frame_id)

    result = []
    for clip in clips:
      result.append({
          'position': clip[0],
          'length': len(clip),
          'rgb_images': [rgb_images[frame_id] for frame_id in clip],
          })
    return result
