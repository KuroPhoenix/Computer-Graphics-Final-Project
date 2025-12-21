from typing import Iterable, Optional
import collections
import threading


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


class CompositePianoSink(PianoEventSink):
    """Fan-out sink that forwards events to multiple sinks."""

    def __init__(self, sinks: Optional[Iterable[PianoEventSink]] = None) -> None:
        self._sinks = [s for s in (sinks or []) if s]

    def add_sink(self, sink: PianoEventSink) -> None:
        if sink and sink not in self._sinks:
            self._sinks.append(sink)

    def handle_note_event(self, event: dict, playback_time_ms: Optional[float] = None) -> None:
        for sink in self._sinks:
            sink.handle_note_event(event, playback_time_ms=playback_time_ms)

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()


class NoteCaptureSink(PianoEventSink):
    """Thread-safe capture of the most recent note event."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_event: Optional[dict] = None
        self._last_playback_time_ms: Optional[float] = None
        self._active_notes: dict[tuple[int, int], int] = {}
        self._event_queue: collections.deque[dict] = collections.deque()

    def handle_note_event(self, event: dict, playback_time_ms: Optional[float] = None) -> None:
        event_type = event.get("type")
        note = event.get("note")
        channel = event.get("channel", 0)
        velocity = event.get("velocity", 0)
        with self._lock:
            if event_type == "on" and velocity > 0:
                self._active_notes[(channel, note)] = int(velocity)
            elif event_type in ("off", "on"):
                self._active_notes.pop((channel, note), None)
            payload = dict(event)
            if playback_time_ms is not None:
                payload["playback_time_ms"] = playback_time_ms
            self._last_event = payload
            self._last_playback_time_ms = playback_time_ms
            self._event_queue.append(payload)

    def snapshot(self) -> tuple[Optional[dict], Optional[float], list[dict]]:
        with self._lock:
            if self._last_event is None:
                return None, None, []
            event = dict(self._last_event)
            active_notes = [
                {"channel": ch, "note": note, "velocity": vel}
                for (ch, note), vel in sorted(self._active_notes.items())
            ]
        return event, self._last_playback_time_ms, active_notes

    def drain(self) -> list[dict]:
        with self._lock:
            if not self._event_queue:
                return []
            events = list(self._event_queue)
            self._event_queue.clear()
        return events
