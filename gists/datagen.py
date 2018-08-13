#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports

import bz2file
import six
import six.moves.urllib_request as urllib  # Imports urllib on Python2, urllib.request on Python3
import re

# Dependency imports

import tensorflow as tf
import mwparserfromhell
from nltk.tokenize import sent_tokenize


flags = tf.flags
FLAGS = flags.FLAGS

# See trainer_utils.py for additional command-line flags.
flags.DEFINE_string("data_dir", "data/hmm_data", "Data directory.")
flags.DEFINE_string("tmp_dir", "data/hmm_tmp",
                    "Temporary storage directory.")
flags.DEFINE_integer("max_docs", 100000,
                    "Number of articles to process into data.")

# input window size (number of words) 
# if number of tokens after encoding is greater than input size, crop off start
INPUT_SIZE = 20
TARGET_SIZE = 3


def split_into_sentences(text):
  sentences = sent_tokenize(text)
  split_sentences = []
  for sentence in sentences:
    for split_sentence in sentence.split('\n'):
      split_sentences.append(split_sentence.strip())
  return split_sentences


def download_report_hook(count, block_size, total_size):
  """Report hook for download progress.
  Args:
    count: current block number
    block_size: block size
    total_size: total size
  """
  percent = int(count * block_size * 100 / total_size)
  print("\r%d%%" % percent + " completed", end="\r")


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory.
  Args:
    directory: path to the directory that will be used.
    filename: name of the file to download to (do nothing if it already exists).
    url: URL to download from.
  Returns:
    The path to the downloaded file.
  """
  if not tf.gfile.Exists(directory):
    tf.logging.info("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not tf.gfile.Exists(filepath):
    tf.logging.info("Downloading %s to %s" % (url, filepath))
    inprogress_filepath = filepath + ".incomplete"
    inprogress_filepath, _ = urllib.urlretrieve(
        url, inprogress_filepath, reporthook=download_report_hook)
    # Print newline to clear the carriage return from the download progress
    print()
    tf.gfile.Rename(inprogress_filepath, filepath)
    statinfo = os.stat(filepath)
    tf.logging.info("Successfully downloaded %s, %s bytes." %
                    (filename, statinfo.st_size))
  else:
    tf.logging.info("Not downloading, file already found: %s" % filepath)
  return filepath

def _maybe_download_corpus(tmp_dir):
  """Download corpus if necessary.
  Args:
    tmp_dir: directory containing dataset.
  Returns:
    filepath of the downloaded corpus file.
  """
  corpus_url = ("https://dumps.wikimedia.org/simplewiki/20171201/"
                "simplewiki-20171201-pages-articles-multistream.xml.bz2")
  corpus_filename = os.path.basename(corpus_url)
  corpus_filepath = os.path.join(tmp_dir, corpus_filename)
  if not tf.gfile.Exists(corpus_filepath):
    maybe_download(tmp_dir, corpus_filename, corpus_url)
  return corpus_filepath

def _page_text(page):
  start_pos = page.find(u"<text")
  end_pos = page.find(u"</text>")

  # this means that there isn't any text to grab, so return emtpy string
  # this applies to a few pages titled, "MediaWiki:[topic]"
  if end_pos == -1:
    return None

  assert start_pos != -1, page
  assert end_pos != -1, page

  start_pos += len(u"<text xml:space='preserve'>")
  return page[start_pos:end_pos]


def clean_text(text):
  cleaned_text = re.sub('\w+\|\w+\|\w+\|\w+', '', text)
  cleaned_text = re.sub('<\w+>[\w|\s]*<\/\w+>', '', cleaned_text)
  cleaned_text = re.sub('[^\w\.\'",@!?;:\-\(\)\s]', '', cleaned_text)
  return cleaned_text

def space_text(text):
  spaced_text = re.sub('\.', ' . ', text)
  spaced_text = re.sub(',', ' , ', spaced_text)
  spaced_text = re.sub(';', ' ; ', spaced_text)
  spaced_text = re.sub(':', ' : ', spaced_text)
  spaced_text = re.sub('\'', ' \' ', spaced_text)
  spaced_text = re.sub('"', ' "" ', spaced_text)
  spaced_text = re.sub('!', ' ! ', spaced_text)
  spaced_text = re.sub('\?', ' ? ', spaced_text)
  spaced_text = re.sub('@', ' @ ', spaced_text)
  spaced_text = re.sub('\(', ' ( ', spaced_text)
  spaced_text = re.sub('\)', ' ) ', spaced_text)
  spaced_text = re.sub('\-', ' - ', spaced_text)
  return spaced_text


def page_generator(tmp_dir, max_docs=None):
  """
  Generate cleaned wikipedia articles as a string.
  """
  doc = u""
  count = 0
  corpus_filepath = _maybe_download_corpus(tmp_dir)
  for line in bz2file.BZ2File(corpus_filepath, "r", buffering=1000000):
    line = unicode(line, "utf-8") if six.PY2 else line.decode("utf-8")
    if not doc and line != u"  <page>\n":
      continue
    doc += line
    if line == u"  </page>\n":
      doc_text = _page_text(doc)
      if doc_text != None:
        parsed_text = mwparserfromhell.parse(doc_text) \
          .strip_code(normalize=True, collapse=True)
        yield parsed_text

      doc = u""
      count += 1
      if max_docs and count >= max_docs:
        break


def window_generator(tmp_dir, max_docs=None):  
  for page in page_generator(tmp_dir, max_docs):
    split_page = space_text(clean_text(page)).split()

    for window_end in range(len(page)):
      window_list = split_page[max(0, window_end - INPUT_SIZE):window_end]
      window = ' '.join(window_list)
      next_word = split_page[window_end]

      print('Window: {}, Next word: {}'.format(window, next_word))
      yield window, next_word


def sentence_writer(tmp_dir, data_dir, max_docs):
  with open(os.path.join(data_dir, 'smaller.input.txt'), 'w+') as file:
    for page in page_generator(tmp_dir, max_docs):
      cleaned_page = clean_text(page)
      sentences = split_into_sentences(cleaned_page)
      for sentence in sentences:
        if len(sentence) > 0:
          # file.write(sentence + "\n")
          spaced_sentence = space_text(sentence)
          file.write(spaced_sentence + "\n")


def token_writer(tmp_dir, data_dir, max_docs):
  train_inputs_filepath = os.path.join(data_dir, 'train.inputs.tok')
  train_targets_filepath = os.path.join(data_dir, 'train.targets.tok')
  dev_inputs_filepath = os.path.join(data_dir, 'dev.inputs.tok')
  dev_targets_filepath = os.path.join(data_dir, 'dev.targets.tok')

  train_inputs_file = open(train_inputs_filepath, 'w+')
  train_targets_file = open(train_targets_filepath, 'w+')
  dev_inputs_file = open(dev_inputs_filepath, 'w+')
  dev_targets_file = open(dev_targets_filepath, 'w+')

  i = 1

  for window, next_word in window_generator(tmp_dir, max_docs):
    if i % 10 == 0:
      dev_inputs_file.write(window.encode('utf-8') + '\n')
      dev_targets_file.write(next_word.encode('utf-8') + '\n')
    else:
      train_inputs_file.write(window.encode('utf-8') + '\n')
      train_targets_file.write(next_word.encode('utf-8') + '\n')

    i += 1

  train_inputs_file.close()
  train_targets_file.close()
  dev_inputs_file.close()
  dev_targets_file.close()

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  sentence_writer(FLAGS.tmp_dir, FLAGS.data_dir, FLAGS.max_docs)
  # token_writer(FLAGS.tmp_dir, FLAGS.data_dir, FLAGS.max_docs)

if __name__ == "__main__":
  tf.app.run()