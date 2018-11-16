#!/usr/bin/env bash
# Run high level unit tests under integration_test/
# Those may take a long time to run 

python -m unittest discover integration_test '*.py'
