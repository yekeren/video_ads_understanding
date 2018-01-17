
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf

from video_predictors.video_predictor import VideoPredictor


def load_vocab(vocab_path):
  """Load vocabulary from file.

  Args:
    vocab_path: path to the vocab file.

  Returns:
    vocab: a list mapping from id to text.
  """
  with open(vocab_path, 'r') as fp:
    lines = fp.readlines()
  vocab = ['<UNK>'] * (len(lines) + 1)
  for index, line in enumerate(lines):
    word = line.strip('\n')
    vocab[index + 1] = word
  return vocab


class ADVISEVideoPredictor(VideoPredictor):
  def __init__(self):
    """Initializes the VideoPredictor."""
    graph_path = 'output/ADVISE.pbtxt'

    self._encoders = [
      {
        'name': 'Statement',
        'vocab': load_vocab('output/glove_w2v.txt'),
        'export_knn_ids': 'STMT_KNN_IDS',
        'export_knn_dists': 'STMT_KNN_DISTS',
        'export_proposal_knn_ids': 'PROPOSAL_STMT_KNN_IDS',
        'export_proposal_knn_dists': 'PROPOSAL_STMT_KNN_DISTS',
      },
      {
        'name': 'Densecap',
        'vocab': load_vocab('output/glove_w2v.txt'),
        'export_knn_ids': 'DENSECAP_KNN_IDS',
        'export_knn_dists': 'DENSECAP_KNN_DISTS',
        'export_proposal_knn_ids': 'PROPOSAL_DENSECAP_KNN_IDS',
        'export_proposal_knn_dists': 'PROPOSAL_DENSECAP_KNN_DISTS',
      },
      {
        'name': 'Symbol',
        'vocab': load_vocab('output/symbols.txt'),
        'export_knn_ids': 'SYMBOL_KNN_IDS',
        'export_knn_dists': 'SYMBOL_KNN_DISTS',
        'export_proposal_knn_ids': 'PROPOSAL_SYMBOL_KNN_IDS',
        'export_proposal_knn_dists': 'PROPOSAL_SYMBOL_KNN_DISTS',
      },
    ]
    self._tensors = {}

    # Load the ADVISE graph.
    advise_graph  = tf.Graph()
    with advise_graph.as_default():
      advise_graph_def = tf.GraphDef()
      with tf.gfile.GFile(graph_path, 'rb') as fp:
        advise_graph_def.ParseFromString(fp.read())
        tf.import_graph_def(advise_graph_def, name='')

        self._image_placeholder = advise_graph.get_tensor_by_name(
            'IMAGE_TENSOR:0')
        self._num_detections = advise_graph.get_tensor_by_name(
            'NUM_DETECTIONS:0')

        for encoder in self._encoders:
          for name, tensor_name in encoder.iteritems():
            if not name in ['vocab', 'name']:
              self._tensors[tensor_name] = advise_graph.get_tensor_by_name(
                  tensor_name + ':0')
              tf.logging.info('%s succ.', name)

    # Initialize session.
    default_config = tf.ConfigProto()
    default_config.allow_soft_placement = True
    default_config.gpu_options.allow_growth = True
    default_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    self._sess = tf.Session(graph=advise_graph, config=default_config) 

  def __del__(self):
    """Release the VideoPredictor."""
    self._sess.close()

  def predict(self, rgb_images):
    """Give a bunch of video frames, predict the info of the video clip.

    Args:
      rgb_images: A list of (image_height, image_width, channes) image frames.
    """
    image = rgb_images[0]
    tensor_vals = self._sess.run(self._tensors, 
        feed_dict={self._image_placeholder: image})

    predictions = []
    for encoder in self._encoders:
      prediction = {
        'name': encoder['name'],
        'results': []
      }
      for word_id, distance in zip(
          tensor_vals[encoder['export_knn_ids']],
          tensor_vals[encoder['export_knn_dists']]):
        prediction['results'].append({
            'word': encoder['vocab'][word_id],
            'score': '%.3lf' % distance
            })
      predictions.append(prediction)
    return predictions
