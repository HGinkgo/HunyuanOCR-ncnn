# Tools

This directory contains model packaging, layout checks, and regression
verification helpers for the C++ runtime.

`package_model.py` builds the standard runtime model directory from exported
workspace artifacts. It creates symlinks by default and supports `--copy` for
portable bundles. Use `--vision-backend fixed`, `dynamic`, or `both` to choose
the packaged vision layout.

`run_example.py` runs one bundled image from `examples/images/` through the
compiled CLI. Use `--list` to show available cases, and `--prompt` to pass a
custom image prompt instead of the case's built-in prompt mode.

`run_examples.py` runs all bundled example images and writes logs to
`outputs/examples/` by default.

`run_regression.py` runs the bundled regression image cases against a
packaged model directory. It requires exported fixture directories from the
baseline/export workflow. Pass `--package` to rebuild the packaged model first;
`--package-vision-backend dynamic` runs the same regression through the dynamic
vision package. Manifest cases may use either a built-in `prompt_mode` or a
literal custom `prompt`.
