# Tests

1. **TDD loop.** Load the `tdd` skill before writing or changing a test here, and follow its loop.
2. **Test at seams.** A test imports and calls only seams. A seam is a public name: no segment of
   its full dotted name where it is defined (`module.Class.method`) starts with `_`. The defining
   path decides, whatever re-exports it: `aklr.core.validation.stages.bench._task._batch.Batch` is
   private. Code behind a seam is tested through the seam. These are the `tdd` skill's pre-agreed
   seams: write the test at one directly; confirm with the user only an interface still undecided.
3. **GPU tests run on a GPU.** Tests marked `gpu` or `gfx950` are skipped on a machine without that
   GPU (`conftest.py`), and a skip is neither red nor green: run them on a GPU, a remote node from the
   `remote-gpu` skill if none is local.
