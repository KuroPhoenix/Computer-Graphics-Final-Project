import logging
import mido
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from configs.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class MidiPlaybackState:
    timeline: List[Dict]
    cursor_idx: int = 0
    status: str = "stopped"  # "stopped", "playing", "paused"
    start_clock: Optional[float] = None
    paused_elapsed_ms: float = 0.0
    duration_ms: int = 0
    tempo_map: List[Dict] = field(default_factory=list)


class MidiManager:
    def __init__(self, clock: Optional[Callable[[], float]] = None, poll_interval_ms: int = 5):
        """
        clock: returns current time in milliseconds.
        poll_interval_ms: scheduler sleep interval when waiting for the next event.
        """
        self._clock = clock or (lambda: time.perf_counter() * 1000)
        self._poll_interval_ms = poll_interval_ms
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def parse_midi(self, path: str) -> List[Dict]:
        """Parse a MIDI file into a time-ordered event list."""
        log.info("Parsing MIDI file: %s", path)
        try:
            mid = mido.MidiFile(path)
        except Exception as exc:
            log.error("Unable to load MIDI file at %s: %s", path, exc)
            raise FileNotFoundError(f"Unable to load MIDI file at {path}: {exc}") from exc

        abs_time = 0.0
        events: List[Dict] = []
        for msg in mid:  # merged, time in seconds
            abs_time += msg.time
            if msg.type == "set_tempo":
                events.append({"time_ms": int(abs_time * 1000), "type": "tempo", "tempo": msg.tempo})
            elif msg.type == "program_change":
                events.append({
                    "time_ms": int(abs_time * 1000),
                    "type": "program_change",
                    "program": msg.program,
                    "channel": msg.channel,
                })
            elif msg.type in ("note_on", "note_off"):
                is_on = msg.type == "note_on" and msg.velocity > 0
                events.append({
                    "time_ms": int(abs_time * 1000),
                    "type": "on" if is_on else "off",
                    "note": msg.note,
                    "velocity": msg.velocity if is_on else 0,
                    "channel": msg.channel,
                })
        events.sort(key=lambda e: e["time_ms"])

        duration_ms = events[-1]["time_ms"] if events else 0
        tempo_events = sum(1 for e in events if e.get("type") == "tempo")
        log.info("Parsed %d events (duration=%d ms, tempo events=%d)", len(events), duration_ms, tempo_events)
        return events

    def load_timeline(self, events: List[Dict]) -> MidiPlaybackState:
        """Create playback state from a parsed timeline (immutable event list)."""
        log.debug("Loading timeline with %d events", len(events))
        duration_ms = events[-1]["time_ms"] if events else 0
        tempo_events = [e for e in events if e.get("type") == "tempo"]
        state = MidiPlaybackState(
            timeline=events,
            cursor_idx=0,
            status="stopped",
            start_clock=None,
            paused_elapsed_ms=0.0,
            duration_ms=duration_ms,
            tempo_map=tempo_events,
        )
        log.info(
            "Initialized playback state (duration=%d ms, tempo events=%d)",
            duration_ms,
            len(tempo_events),
        )
        return state

    def start_playback(
        self,
        state: MidiPlaybackState,
        on_event: Callable[[Dict], None],
        on_end: Optional[Callable[[], None]] = None,
    ) -> None:
        """Start playback in a background thread, dispatching events to on_event."""
        with self._lock:
            if state.status == "playing":
                log.warning("Playback already running")
                return
            if state.cursor_idx >= len(state.timeline):
                state.cursor_idx = 0
                state.paused_elapsed_ms = 0.0
            if self._thread and self._thread.is_alive():
                log.debug("Waiting for previous playback thread to finish")
                self._stop_event.set()
                self._thread.join(timeout=1)
                self._stop_event.clear()

            self._stop_event.clear()
            state.start_clock = self._clock()
            state.status = "playing"
            self._thread = threading.Thread(
                target=self._run_scheduler, args=(state, on_event, on_end), daemon=True
            )
            self._thread.start()
            log.info("Playback started")

    def pause(self, state: MidiPlaybackState) -> None:
        """Pause playback, retaining position."""
        with self._lock:
            if state.status != "playing":
                log.debug("Pause ignored; status=%s", state.status)
                return
            state.paused_elapsed_ms = self._elapsed_ms(state)
            state.start_clock = None
            state.status = "paused"
            log.info("Playback paused at %d ms", state.paused_elapsed_ms)

    def resume(self, state: MidiPlaybackState) -> None:
        """Resume playback from paused position."""
        with self._lock:
            if state.status != "paused":
                log.debug("Resume ignored; status=%s", state.status)
                return
            state.start_clock = self._clock()
            state.status = "playing"
            log.info("Playback resumed at %d ms", state.paused_elapsed_ms)

    def stop(self, state: MidiPlaybackState) -> None:
        """Stop playback and reset to start."""
        with self._lock:
            state.status = "stopped"
            state.start_clock = None
            state.paused_elapsed_ms = 0.0
            state.cursor_idx = 0
            self._stop_event.set()
            log.info("Playback stopped and reset")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._stop_event.clear()

    def seek(self, state: MidiPlaybackState, target_ms: int) -> None:
        """Seek to target_ms and update cursor; keeps current play/pause status."""
        target_ms = max(0, min(target_ms, state.duration_ms))
        with self._lock:
            state.cursor_idx = self._find_cursor(state.timeline, target_ms)
            state.paused_elapsed_ms = target_ms
            if state.status == "playing":
                state.start_clock = self._clock()
            else:
                state.start_clock = None
            log.info("Seeked to %d ms (cursor=%d)", target_ms, state.cursor_idx)

    def current_position_ms(self, state: MidiPlaybackState) -> float:
        """Return current playback position in ms."""
        with self._lock:
            return self._elapsed_ms(state)

    # Internal helpers
    def _run_scheduler(
        self,
        state: MidiPlaybackState,
        on_event: Callable[[Dict], None],
        on_end: Optional[Callable[[], None]],
    ) -> None:
        last_debug_log = 0.0
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    status = state.status
                if status == "paused":
                    time.sleep(self._poll_interval_ms / 1000.0)
                    continue
                if status != "playing":
                    break

                now_ms = self._elapsed_ms(state)
                dispatched = 0
                while state.cursor_idx < len(state.timeline) and state.timeline[state.cursor_idx]["time_ms"] <= now_ms:
                    event = state.timeline[state.cursor_idx]
                    state.cursor_idx += 1
                    dispatched += 1
                    try:
                        on_event(event)
                    except Exception as exc:
                        log.exception("Error dispatching event %s: %s", event, exc)
                if log.isEnabledFor(logging.DEBUG):
                    now = time.perf_counter()
                    if now - last_debug_log >= 1.0:
                        log.debug(
                            "Scheduler heartbeat: now_ms=%d, cursor=%d/%d, status=%s",
                            now_ms,
                            state.cursor_idx,
                            len(state.timeline),
                            state.status,
                        )
                        last_debug_log = now
                if state.cursor_idx >= len(state.timeline):
                    log.info("Playback reached end of timeline")
                    break
                if dispatched == 0:
                    time.sleep(self._poll_interval_ms / 1000.0)
            # Finalize state on exit
            with self._lock:
                state.paused_elapsed_ms = min(self._elapsed_ms(state), state.duration_ms)
                state.start_clock = None
                if state.cursor_idx >= len(state.timeline):
                    state.status = "stopped"
                elif state.status == "playing":
                    state.status = "stopped"
        finally:
            if on_end:
                try:
                    on_end()
                except Exception as exc:
                    log.exception("Error in on_end callback: %s", exc)

    def _elapsed_ms(self, state: MidiPlaybackState) -> float:
        """Compute current elapsed playback time."""
        if state.start_clock is None:
            return state.paused_elapsed_ms
        return state.paused_elapsed_ms + (self._clock() - state.start_clock)

    @staticmethod
    def _find_cursor(timeline: List[Dict], target_ms: int) -> int:
        """Find the first event index at or after target_ms."""
        for idx, event in enumerate(timeline):
            if event["time_ms"] >= target_ms:
                return idx
        return len(timeline)
