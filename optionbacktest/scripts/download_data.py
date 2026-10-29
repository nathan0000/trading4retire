"""Direct entry point for provider-backed historical downloads."""

import sys

from spx_0dte_backtest.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["download-data", *sys.argv[1:]]))
