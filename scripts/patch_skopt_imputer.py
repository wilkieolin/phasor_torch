#!/usr/bin/env python
"""Make dh-scikit-optimize's SimpleImputers compatible with modern numpy/sklearn.

ytopt's skopt fork builds `SimpleImputer(strategy="constant", ...)` in
`skopt/space/space.py` without `keep_empty_features`. On numpy-2 / recent
sklearn this DROPS all-missing feature columns, collapsing the transformed
space matrix to 0 columns -> "attempt to get argmax of an empty sequence" /
"index 0 is out of bounds for axis 1 with size 0" inside `Optimizer.tell()`.

Setting `keep_empty_features=True` preserves the columns and the matrix shape
skopt expects. This edits the installed fork in place; it is idempotent and
safe to re-run. Run it (in the env where ytopt is installed) after installing
the fork:  `python scripts/patch_skopt_imputer.py`
"""

import os
import sys

import skopt

PATH = os.path.join(os.path.dirname(skopt.__file__), "space", "space.py")

OLD_NEW = [
    ('strategy="constant", fill_value=-1000\n        )',
     'strategy="constant", fill_value=-1000, keep_empty_features=True\n        )'),
    ('strategy="constant", fill_value=np.nan\n        )',
     'strategy="constant", fill_value=np.nan, keep_empty_features=True\n        )'),
]


def main() -> int:
    with open(PATH) as fh:
        src = fh.read()
    if "keep_empty_features" in src:
        print(f"already patched: {PATH}")
        return 0
    patched = src
    for old, new in OLD_NEW:
        patched = patched.replace(old, new)
    if patched == src:
        print("ERROR: imputer pattern not found; fork layout may have changed",
              file=sys.stderr)
        return 1
    with open(PATH, "w") as fh:
        fh.write(patched)
    print(f"patched {PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
