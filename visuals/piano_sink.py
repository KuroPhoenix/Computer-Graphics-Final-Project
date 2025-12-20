from typing import Optional


class PianoEventSink:
    """Interface for consumers that render/animate a virtual piano."""

    def handle_note_event(self, event: dict, playback_time_ms: Optional[float] = None) -> None:
        """Consume a note on/off event."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources."""
        return


class NullPianoSink(PianoEventSink):
    """No-op sink used when no renderer is attached."""

    def handle_note_event(self, event: dict, playback_time_ms: Optional[float] = None) -> None:
        return
