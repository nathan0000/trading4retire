from pathlib import Path

from spx_0dte_backtest.cli import main

ROOT = Path(__file__).parents[2]


def test_option_validation_cli_accepts_synthetic_example(capsys) -> None:
    code = main(
        [
            "validate-options",
            "--input",
            str(ROOT / "data/examples/spx_0dte_options_example.csv"),
            "--config",
            str(ROOT / "config/settings.yaml"),
            "--json",
        ]
    )
    assert code == 0
    assert '"accepted_rows": 4' in capsys.readouterr().out
