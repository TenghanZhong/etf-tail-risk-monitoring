"""Entry point for the ETF tail-risk monitoring experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import ClassVar

from src import tail_risk_monitoring_core as _core
from src.tail_risk_monitoring_core import *  # noqa: F401,F403


@dataclass
class Config(_core.Config):
    """Repository-level configuration for the main experiment."""

    output_dir_name: str = "results"
    base_dir: ClassVar[Path] = Path(
        os.environ.get("ETF_TAIL_RISK_DATA_DIR", Path(__file__).resolve().parent)
    )


_core.Config = Config


def main() -> None:
    """Run the main monitoring experiment and its ablations."""

    _core.main()


if __name__ == "__main__":
    main()
