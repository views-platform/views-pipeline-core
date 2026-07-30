"""Temporal partition domain value objects.

``TemporalPartition`` represents a contiguous month range (inclusive).
``PartitionSet`` bundles a train and test partition with non-overlap
validation at construction.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalPartition:
    """A contiguous, inclusive month range [start, end].

    Attributes:
        start: First month_id (inclusive).
        end: Last month_id (inclusive).
    """

    start: int
    end: int

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(
                f"TemporalPartition start ({self.start}) > end ({self.end})"
            )

    @property
    def base_origin(self) -> int:
        """Month before partition start — anchor for rolling-origin evaluation."""
        return self.start - 1

    @property
    def length(self) -> int:
        """Number of months in the partition (inclusive)."""
        return self.end - self.start + 1


@dataclass(frozen=True)
class PartitionSet:
    """A train/test partition pair with non-overlap guarantee.

    Validates at construction that ``train.end < test.start``.
    """

    train: TemporalPartition
    test: TemporalPartition

    def __post_init__(self):
        if self.train.end >= self.test.start:
            raise ValueError(
                f"Train partition end ({self.train.end}) overlaps with "
                f"test partition start ({self.test.start}). "
                f"Train must end before test begins."
            )

    @classmethod
    def from_dict(cls, d: dict, partition_key: str) -> "PartitionSet":
        """Parse from legacy ``partition_dict`` format.

        Expects ``d[partition_key]`` to contain ``"train"`` and ``"test"``
        keys, each mapping to a ``(first_month, last_month)`` tuple.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If partitions overlap or are inverted.
        """
        sub = d[partition_key]
        return cls(
            train=TemporalPartition(*sub["train"]),
            test=TemporalPartition(*sub["test"]),
        )
