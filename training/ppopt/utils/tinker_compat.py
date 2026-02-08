"""Compatibility helpers for Tinker SDK on older Python versions."""

import enum


def ensure_tinker_compat() -> None:
    """Patch enum.StrEnum on Python < 3.11 so tinker can import."""
    if not hasattr(enum, "StrEnum"):
        class StrEnum(str, enum.Enum):
            pass
        enum.StrEnum = StrEnum  # type: ignore[attr-defined]
