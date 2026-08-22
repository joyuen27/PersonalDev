#!/bin/bash
set -e
cd "$BUILD_WORKSPACE_DIRECTORY"
python3 model/gen_tokens.py
bazel-bin/infer_bin
