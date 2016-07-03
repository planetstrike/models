#!/bin/bash
# A script that runs a tokenizer, a part-of-speech tagger and a dependency
# parser on an English text file, with one sentence per line.
#
# Example usage:
#  echo "Parsey McParseface is my favorite parser!" | syntaxnet/demo.sh

# To run on a conll formatted file, add the --conll command line argument.
#

PARSER_EVAL=bazel-bin/syntaxnet/parser_eval
MODEL_DIR=syntaxnet/models/parsey_mcparseface
[[ "$1" == "--conll" ]] && INPUT_FORMAT=stdin-conll || INPUT_FORMAT=stdin

$PARSER_EVAL \
  --input=$INPUT_FORMAT \
  --output=stdout-conll \
  --hidden_layer_sizes=64 \
  --arg_prefix=brain_tagger \
  --graph_builder=structured \
  --task_context=$MODEL_DIR/context.pbtxt \
  --model_path=$MODEL_DIR/tagger-params \
  --slim_model \
  --batch_size=1024 \
  --alsologtostderr \
   | \
  $PARSER_EVAL \
  --input=stdin-conll \
  --output=stdout-conll \
  --hidden_layer_sizes=512,512 \
  --arg_prefix=brain_parser \
  --graph_builder=structured \
  --task_context=$MODEL_DIR/context.pbtxt \
  --model_path=$MODEL_DIR/parser-params \
  --slim_model \
  --batch_size=1024 \
  --alsologtostderr \
  | \
  bazel-bin/syntaxnet/conll2xml \
  --task_context=$MODEL_DIR/context.pbtxt \
  --alsologtostderr
