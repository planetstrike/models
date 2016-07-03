"""A program to generate an XML document from conll files."""

import collections

import dicttoxml
import tensorflow as tf

import syntaxnet.load_parser_ops

from tensorflow.python.platform import tf_logging as logging
from syntaxnet import sentence_pb2
from syntaxnet.ops import gen_parser_ops
from xml.dom.minidom import parseString

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('task_context',
                    'syntaxnet/models/parsey_mcparseface/context.pbtxt',
                    'Path to a task context with inputs and parameters for '
                    'feature extractors.')
flags.DEFINE_string('corpus_name', 'stdin-conll',
                    'Path to a task context with inputs and parameters for '
                    'feature extractors.')


def to_dict(sentence):
  """Builds a dictionary representing the parse tree of a sentence.

  Args:
    sentence: Sentence protocol buffer to represent.
  Returns:
    Dictionary mapping tokens to children.
  """
  token_str = ['%s %s %s' % (token.word, token.tag, token.label)
               for token in sentence.token]
  children = [[] for token in sentence.token]
  root = -1
  for i in range(0, len(sentence.token)):
    token = sentence.token[i]
    if token.head == -1:
      root = i
    else:
      children[token.head].append(i)

  def _get_dict(i):
    d = collections.OrderedDict()
    for c in children[i]:
      d[token_str[c]] = _get_dict(c)
    return d

  tree = collections.OrderedDict()
  tree[token_str[root]] = _get_dict(root)
  return tree


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  with tf.Session() as sess:
    src = gen_parser_ops.document_source(batch_size=32,
                                         corpus_name=FLAGS.corpus_name,
                                         task_context=FLAGS.task_context)
    sentence = sentence_pb2.Sentence()
    while True:
      documents, finished = sess.run(src)
      logging.info('Read %d documents', len(documents))
      for d in documents:
        sentence.ParseFromString(d)
        # tr = asciitree.LeftAligned()
        d = to_dict(sentence)
        print 'Input: %s' % sentence.text
        print 'Parse:'
	dom = parseString(dicttoxml.dicttoxml(d, attr_type=False))
	print dom.toprettyxml()
        #print dicttoxml.dicttoxml(d)

      if finished:
        break


if __name__ == '__main__':
  tf.app.run()
