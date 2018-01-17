
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import logging

import feature_extractor
from video_predictors.video_predictor import VideoPredictor


def load_yt8m_vocab(filename):
  """Loads youtube-8m vocabulary file.

  Args:
    filename: Path to vocabulary file downloaded from youtube-8m website.

  Returns:
    id_to_name: A python dict mapping from integer id to class name.
  """
  with open(filename, 'r') as fp:
    lines = fp.readlines()

  id_to_name = {}
  for line in lines[1:]:
    id_str, _, _, name_str = line.strip('\n').split(',')[:4]
    id_to_name[int(id_str)] = name_str.lower()

  logging.info('Load %i classes from vocabulary.', len(id_to_name))
  return id_to_name


class Youtube8MVideoPredictor(VideoPredictor):
  def __init__(self):
    """Initializes the VideoPredictor."""
    # Load vocabulary.
    self._id_to_name = load_yt8m_vocab(
        '/afs/cs.pitt.edu/usr0/yekeren/work2/video_ads_understanding/youtube-8m/ads_features/vocabulary.csv')

    # Initialize image feature extractor.
    self._extractor = feature_extractor.YouTube8MFeatureExtractor(
        '/afs/cs.pitt.edu/usr0/yekeren/work2/video_ads_understanding/youtube-8m/yt8m')

    # Find the latest checkpoint.
    latest_checkpoint = '/afs/cs.pitt.edu/usr0/yekeren/work2/video_ads_understanding/youtube-8m/models/video_level_logistic_model/model.ckpt-23010'
    meta_graph_location = latest_checkpoint + ".meta"
    logging.info("loading meta-graph: " + meta_graph_location)

    yt8m_graph = tf.Graph()
    with yt8m_graph.as_default():
      saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
      self._input_tensor = tf.get_collection("input_batch_raw")[0]
      self._num_frames_tensor = tf.get_collection("num_frames")[0]
      self._predictions_tensor = tf.get_collection("predictions")[0]

    # Initialize session.
    default_config = tf.ConfigProto()
    default_config.allow_soft_placement = True
    default_config.gpu_options.allow_growth = True
    default_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    self._sess = tf.Session(graph=yt8m_graph, config=default_config) 

    logging.info("restoring variables from " + latest_checkpoint)
    saver.restore(self._sess, latest_checkpoint)

  def __del__(self):
    """Release the VideoPredictor."""
    self._sess.close()

  def predict(self, rgb_images):
    """Give a bunch of video frames, predict the info of the video clip.

    Args:
      rgb_images: A list of (image_height, image_width, channes) image frames.
    """
    rgb_features = [self._extractor.extract_rgb_frame_features(
        rgb) for rgb in rgb_images]

    rgb_features = np.stack(rgb_features, axis=0)
    mean_rgb = rgb_features.mean(axis=0)

    predictions_val, = self._sess.run([self._predictions_tensor], 
        feed_dict={ 
          self._input_tensor: [mean_rgb], 
          self._num_frames_tensor: [1] 
        })
    top_k = 10
    top_indices = np.argsort(-predictions_val[0])[:top_k]

    # Wrap the predictions.
    prediction = {
      'name': 'Yt8m',
      'results': []
    }
    for word_id in top_indices:
      score = predictions_val[0, word_id]
      prediction['results'].append({
          'word': self._id_to_name[word_id],
          'score': '%.3lf' % score
          })
    return [prediction]
