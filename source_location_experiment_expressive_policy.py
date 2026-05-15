"""
Backward-compatibility shim.

The code previously in this file has been moved to::

    boedx/experiments/source_location.py

All public names are re-exported here.  The CLI entry point (``main()``)
is also preserved — running this file directly still works.

Preferred usage going forward::

    # Install the package
    pip install -e .

    # Run via the installed console script
    boedx-source-location --help

    # Or as a module
    python -m boedx.experiments.source_location --help
"""

# ruff: noqa: F401

from boedx.experiments.source_location import (
    VARIANTS,
    SourceLocationConfig,
    SourceLocalization2DEnv,
    _canonical_pair_numpy,
    _canonical_pair_torch,
    main,
    parse_args,
)

if __name__ == "__main__":
    main()
