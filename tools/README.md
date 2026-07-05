# Tools

This directory is reserved for model packaging, layout checks, and regression
verification helpers used by the C++ runtime.

`package_model.py` builds the standard runtime model directory from exported
workspace artifacts. It creates symlinks by default and supports `--copy` for
portable bundles.

`run_5sample_regression.py` runs the current five golden image cases against a
packaged model directory. Pass `--package` to rebuild the packaged model first.
