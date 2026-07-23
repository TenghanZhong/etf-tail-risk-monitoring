"""Entry point for the focused input-quality validation experiment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import ClassVar

from src import quality_layer_core as _core
from src.quality_layer_core import *  # noqa: F401,F403


@dataclass
class Config(_core.Config):
    """Repository-level configuration for the quality experiment."""

    output_dir_name: str = "results_quality_validation"
    base_dir: ClassVar[Path] = Path(
        os.environ.get("ETF_TAIL_RISK_DATA_DIR", Path(__file__).resolve().parent)
    )


_core.Config = Config


def main() -> None:
    """Run the prediction-time input-quality validation experiment."""

    _core.main()


if __name__ == "__main__":
    main()
