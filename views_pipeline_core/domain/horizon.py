"""Forecast horizon domain value object.

Encodes the rolling-origin evaluation parameters: how many steps ahead
the model forecasts, stride between origins, and the maximum shift count.
Provides ``build_step_mappings()`` as the single source of truth for
month_id → lead-time mappings (consolidates C-39 duplication).
"""
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ForecastHorizon:
    """Rolling-origin evaluation parameters.

    Attributes:
        time_steps: Number of months forecast ahead (default 36).
        stride: Step between consecutive rolling origins (default 1).
        max_shift_count: Maximum origin shift from base (default 12).
    """

    time_steps: int = 36
    stride: int = 1
    max_shift_count: int = 12

    @property
    def n_sequences(self) -> int:
        """Number of rolling-origin evaluation sequences."""
        return self.max_shift_count + 1

    @property
    def required_test_length(self) -> int:
        """Minimum test partition length for full rolling-origin evaluation."""
        return self.time_steps + self.max_shift_count

    def build_step_mappings(self, base_origin: int) -> List[Dict[int, int]]:
        """Build month_id -> step mappings for all rolling-origin sequences.

        Args:
            base_origin: The month before the test partition starts
                (typically ``test_partition.start - 1``).

        Returns:
            List of dicts, one per sequence. Each dict maps
            ``{month_id: lead_time_step}`` for that origin.
        """
        return [
            {
                base_origin + i + s: s
                for s in range(self.time_steps)
            }
            for i in range(self.n_sequences)
        ]

    @classmethod
    def default(cls) -> "ForecastHorizon":
        """Production defaults: 36 steps, stride 1, 12 max shifts."""
        return cls(time_steps=36, stride=1, max_shift_count=12)
