"""Entry point for the GJR-GARCH benchmark comparison."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import ClassVar

from src import gjr_garch_comparison_core as _core
from src.gjr_garch_comparison_core import *  # noqa: F401,F403


@dataclass
class Config(_core.Config):
    """Repository-level configuration for the benchmark experiment."""

    output_dir_name: str = "results_gjr_garch"
    base_dir: ClassVar[Path] = Path(
        os.environ.get("ETF_TAIL_RISK_DATA_DIR", Path(__file__).resolve().parent)
    )


_core.Config = Config


def main() -> None:
    """Run the monitoring experiment with the GJR-GARCH comparison."""

    _core.main()


if __name__ == "__main__":
    main()
