#!/bin/bash

# exit immediately upon non-zero exit status
set -e

# create and train model and capture output
# this will create a model pack based on some data included within the tests/resources/regression/creation/ folder,
# it will then train on some self-supervised as well as supervised training data and save the model.
output=$(python tests/backwards_compatibility/creation/cat_creation.py)
# make sure the user sees the output
echo "$output"

# extract the last line of the output which contains the full model path
model_path=$(echo "$output" | tail -n 1)
# NOTE: this file should be tagged with the python version we're using

# test the vocab to make sure it's all good
python tests/backwards_compatibility/testing/test_vocab.py
# TODO: test other things as well?

# run the regression_checker with the captured file path
# if any of the regression cases fail, this will return a non-zero exit status
python -m medcat.utils.regression.regression_checker \
  "$model_path" \
  tests/backwards_compatibility/testing/example_regression_suite.yml \
  --strictness STRICTEST \
  --require-fully-correct

# Step 4: Clean up the generated file(s)
rm -rf "$model_path"*