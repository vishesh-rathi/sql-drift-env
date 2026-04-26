"""P0 smoke — verifies package skeleton imports cleanly.

Deleted / rewritten once concrete modules exist in P1+.
"""


def test_package_imports() -> None:
    import actors
    import engine
    import scenarios
    import skill_library
    import training


def test_version_is_pep440() -> None:
    import re

    import sql_drift_env

    assert re.fullmatch(r"\d+\.\d+\.\d+", sql_drift_env.__version__)
