"""Public Kitsune-TTS API with lazy optional dependencies."""

from typing import TYPE_CHECKING

__all__ = ["KitsuneSynthesizer", "KitsuneTrainer"]

if TYPE_CHECKING:
    from kitsune.api import KitsuneSynthesizer
    from kitsune.trainer import KitsuneTrainer


def __getattr__(name):
    if name == "KitsuneSynthesizer":
        from kitsune.api import KitsuneSynthesizer

        globals()[name] = KitsuneSynthesizer
        return KitsuneSynthesizer
    if name == "KitsuneTrainer":
        from kitsune.trainer import KitsuneTrainer

        globals()[name] = KitsuneTrainer
        return KitsuneTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
