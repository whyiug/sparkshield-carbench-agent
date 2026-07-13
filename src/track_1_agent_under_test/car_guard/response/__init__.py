"""Natural user-facing response generation."""

from .composer import ResponseComposer
from .tts_guard import TTSGuard, TTSViolation

__all__ = ["ResponseComposer", "TTSGuard", "TTSViolation"]
