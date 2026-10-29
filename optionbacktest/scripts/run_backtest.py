"""Direct entry point for the end-to-end backtest workflow."""

import sys

from spx_0dte_backtest.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run-backtest", *sys.argv[1:]]))
