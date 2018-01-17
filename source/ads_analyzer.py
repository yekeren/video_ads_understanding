
import csv
import os
import sys
import json

import cv2
import numpy
import tensorflow as tf
from scipy.spatial import distance
from tensorflow import app
from tensorflow import flags
from tensorflow import logging

from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import frame_iterators.builder as frame_iterators_builder
import clip_iterators.builder as clip_iterators_builder
import video_predictors.builder as video_predictors_builder

FLAGS = flags.FLAGS

flags.DEFINE_string("advise_graph", "",
                    "The path to the proto file of the freezed ADVISE model.")

flags.DEFINE_integer('frames_per_second', 10,
                     'This many frames per second will be processed')

flags.DEFINE_integer('sample_every_n', 10,
                     'Sample one frame from this many frames.')

flags.DEFINE_string('input_videos_csv', None,
                    'CSV file with lines "<video_file>,<labels>", where '
                    '<video_file> must be a path of a video and <labels> '
                    'must be an integer list joined with semi-colon ";"')

flags.DEFINE_string('frame_iterator', 'VideoCaptureFrameIterator',
                    'The name of the frame iterator.')

flags.DEFINE_string('clip_iterator', 'DenseOpticalFlowClipIterator',
                    'The name of the clip iterator.')

flags.DEFINE_integer("top_k", 10,
                     "How many predictions to output per video clip.")

flags.DEFINE_float('distance_threshold', 0.8,
                   'The threshold used to split video clips.')

flags.DEFINE_string("vocab_file", "",
                    "Path to the vocabulary CSV file from youtube-8m.")

flags.DEFINE_string("export_path", "",
                    "The path used to store prediction results.")

flags.DEFINE_string("tmp_dir", "",
                    "The directory used to store visualization results.")

flags.DEFINE_string("partition", "0/1", "")

flags.DEFINE_string("task", "visualize", "")


def _downsample(clips, sample_every_n=1):
  """Downsamples video clips."""
  pos = 0
  frame_id = 0
  downsampled_clips = []
  for clip in clips:
    frames = []
    for rgb_image in clip['rgb_images']:
      if frame_id % sample_every_n == 0:
        frames.append(rgb_image)
      frame_id += 1

    if len(frames) > 0:
      downsampled_clips.append({
          'position': pos, 
          'length': len(frames), 
          'rgb_images': frames
          })
      pos += len(frames)

  return downsampled_clips


def analyze_video(video_file, frame_iterator, clip_iterator, video_predictors,
    lda_model, lda_vectorizer):
  """Uses yt8m model to analyze video clips of the video file.

  Args:
    video_file: Path to video file (e.g. mp4)
    frame_iterator: An instance of FrameIterator.
    clip_iterator: An instance of ClipIterator.
    video_predictors: A list of VideoPredictors.

  Returns:
    A python dict involving the results.
  """
  # Decode video frames from the raw video file.
  rgb_images = []
  for rgb in frame_iterator.frame_iterator(
      video_file, every_ms=1000.0/FLAGS.frames_per_second):
    rgb_images.append(rgb)

  if not rgb_images:
    logging.warning('Could not get features for %s.', video_file)
    return None

  # Split video frames into video clips.
  video_id = video_file.split('/')[-1].split('.')[0]
  clips = [clip for clip in clip_iterator.clip_iterator(rgb_images)]

  # Downsample the video clips.
  if FLAGS.sample_every_n > 1:
    clips = _downsample(clips, FLAGS.sample_every_n)

  # Predict results.
  for clip in clips:
    predictions = []
    for predictor in video_predictors:
      predictions.extend(predictor.predict(clip['rgb_images']))
    clip['predictions'] = predictions

    # LDA.
    words = set()
    for prediction in clip['predictions']:
      if prediction['name'].lower() == 'symbol':
        continue
      for result in prediction['results']:
        if result['word']:
          words.add(result['word'])

    document_word_mat = lda_vectorizer.transform([list(words)])
    document_topic_mat = lda_model.transform(document_word_mat)
    topic_word_mat = lda_model.components_ / lda_model.components_.sum(axis=1)[:, numpy.newaxis]

    document_word_reconstruct = numpy.matmul(document_topic_mat, topic_word_mat)

    lda_vocab = lda_vectorizer.get_feature_names()
    lda_words = [lda_vocab[x] for x in document_word_reconstruct.argsort()[0][::-1][:5]]
    clip['lda_words'] = lda_words

  return {'video_id': video_id, 'clips': clips}


def image_uint8_to_base64(image, ext='.jpg', 
    disp_size=None, convert_to_bgr=False):

  if convert_to_bgr:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if disp_size is not None:
    image = cv2.resize(image, disp_size)
  _, encoded = cv2.imencode(ext, image)
  return encoded.tostring().encode('base64').replace('\n', '')


def visualize(result):
  """Visualizes the result of analysis.

  Args:
    result: A python dict returned by analyze_video.
  """
  video_id = result['video_id']
  clips = result['clips']

  filename = os.path.join(FLAGS.tmp_dir, '%s.html' % (video_id))

  with open(filename, 'w') as fp:
    fp.write('<h1>%s</h1>' % (video_id))
    fp.write('<table border=1>')

    # Print the header.
    fp.write('<tr>')
    fp.write('<th>clip id</th>')
    fp.write('<th>position</th>')
    fp.write('<th>length</th>')
    for prediction in clips[0]['predictions']:
      fp.write('<th>%s prediction</th>' % (prediction['name']))
    fp.write('<th>images</th>')
    fp.write('</tr>')

    # Visualize video clips.
    for clip_id, clip in enumerate(clips):
      fp.write('<tr>')
      fp.write('<td>%i</td>' % clip_id)
      fp.write('<td>%i</td>' % clip['position'])
      fp.write('<td>%i</td>' % clip['length'])

      # Predictions.
      for prediction in clip['predictions']:
        fp.write('<td>')
        for result in prediction['results']:
          if result['word'] in clip['lda_words']:
            fp.write('<p style="background-color:Yellow">%s %s</p>' % (result['score'], result['word']))
          else:
            fp.write('<p>%s %s</p>' % (result['score'], result['word']))
        fp.write('</td>')

      # Frames in the clip.
      fp.write('<td>')
      for rgb in clip['rgb_images']:
        base64_str = image_uint8_to_base64(rgb, 
            disp_size=(100, 100), convert_to_bgr=True)
        fp.write('<img src="data:image/jpg;base64,%s">' % base64_str)
      fp.write('</td>')

      fp.write('</tr>')
    fp.write('<table/>')


def export(result):
  for clip in result['clips']:
    del clip['rgb_images']

  with open(FLAGS.export_path, 'a') as fp:
    print >> fp, json.dumps(result)

def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  frame_iterator = frame_iterators_builder.build(FLAGS.frame_iterator)
  clip_iterator = clip_iterators_builder.build(FLAGS.clip_iterator)
  yt8m_predictor= video_predictors_builder.build('Youtube8MVideoPredictor')
  advise_predictor= video_predictors_builder.build('ADVISEVideoPredictor')

  # Load LDA model.
  with open('output/lda_vocab.txt', 'r') as fp:
    lda_vocab = [line.strip('\n') for line in fp.readlines()]
  vectorizer = CountVectorizer(analyzer=lambda x: x, vocabulary=lda_vocab)
  vectorizer.fit([])
  logging.info('Vocab size: %i.', len(vectorizer.get_feature_names()))

  lda = joblib.load('output/lda.pkl')

  a, b = map(int, FLAGS.partition.split('/'))

  # Analyze video contents.
  total_analyzed = 0
  for video_file, labels in csv.reader(open(FLAGS.input_videos_csv)):
    total_analyzed += 1

    if total_analyzed % b == a:
      logging.info('On video [%i] %s.', total_analyzed, video_file)

      result = analyze_video(video_file, 
          frame_iterator, clip_iterator, [yt8m_predictor, advise_predictor],
          lda, vectorizer)
      if result is not None:
        if FLAGS.task == 'visualize':
          visualize(result)
        else:
          export(result)
    if total_analyzed >= 10:
      break

  logging.info('Done')

if __name__ == '__main__':
  app.run(main)
