import os
import sys

# On macOS torch bundles its own libomp while xgboost loads Homebrew's; with both in one
# process, multi-threaded OpenMP regions segfault whichever library runs second. The
# tests are small, so run them single-threaded. Set before torch or xgboost is imported.
if sys.platform == 'darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
