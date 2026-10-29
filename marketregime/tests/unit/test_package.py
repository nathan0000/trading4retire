"""Sanity check that the package installs and imports without network/DB/broker dependencies."""

import regime_engine


def test_package_importable() -> None:
    assert regime_engine.__doc__ is not None
