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
    duration_ms: float = 0.0
    tempo_map: List[Dict] = field(default_factory=list)
    playback_id: int = 0


class MidiManager:
    def __init__(
        self,
        clock: Optional[Callable[[], float]] = None,
        poll_interval_ms: float = 5.0,
        dispatch_ahead_ms: float = 0.0,
        spin_wait_ms: float = 0.5,
    ):
        """
        clock: returns current time in milliseconds.
        poll_interval_ms: scheduler sleep interval when paused/idle.
        dispatch_ahead_ms: dispatch events early to compensate output latency.
        spin_wait_ms: busy-wait window to improve timing precision.
        """
        self._clock = clock or (lambda: time.perf_counter() * 1000)
        self._poll_interval_ms = poll_interval_ms
        self._dispatch_ahead_ms = dispatch_ahead_ms
        self._spin_wait_ms = spin_wait_ms
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._wake_event = threading.Event()

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
        event_seq = 0
        event_priority = {
            "tempo": 0,
            "control_change": 1,
            "program_change": 2,
            "pitchwheel": 3,
            "channel_pressure": 4,
            "poly_aftertouch": 4,
            "off": 5,
            "on": 6,
        }
        for msg in mid:  # merged, time in seconds
            abs_time += msg.time
            time_ms = abs_time * 1000.0
            event: Optional[Dict] = None
            if msg.type == "set_tempo":
                event = {"time_ms": time_ms, "type": "tempo", "tempo": msg.tempo}
            elif msg.type == "program_change":
                event = {
                    "time_ms": time_ms,
                    "type": "program_change",
                    "program": msg.program,
                    "channel": msg.channel,
                }
            elif msg.type == "control_change":
                event = {
                    "time_ms": time_ms,
                    "type": "control_change",
                    "control": msg.control,
                    "value": msg.value,
                    "channel": msg.channel,
                }
            elif msg.type in ("channel_pressure", "aftertouch"):
                event = {
                    "time_ms": time_ms,
                    "type": "channel_pressure",
                    "pressure": getattr(msg, "value", 0),
                    "channel": msg.channel,
                }
            elif msg.type == "polytouch":
                event = {
                    "time_ms": time_ms,
                    "type": "poly_aftertouch",
                    "note": msg.note,
                    "pressure": msg.value,
                    "channel": msg.channel,
                }
            elif msg.type == "pitchwheel":
                event = {
                    "time_ms": time_ms,
                    "type": "pitchwheel",
                    "pitch": msg.pitch,  # -8192..8191
                    "channel": msg.channel,
                }
            elif msg.type in ("note_on", "note_off"):
                is_on = msg.type == "note_on" and msg.velocity > 0
                event = {
                    "time_ms": time_ms,
                    "type": "on" if is_on else "off",
                    "note": msg.note,
                    "velocity": msg.velocity if is_on else 0,
                    "channel": msg.channel,
                }
            if event:
                event["_seq"] = event_seq
                event["_order"] = event_priority.get(event["type"], 10)
                events.append(event)
                event_seq += 1
        events.sort(key=lambda e: (e["time_ms"], e.get("_order", 10), e.get("_seq", 0)))
        for ev in events:
            ev.pop("_order", None)
            ev.pop("_seq", None)

        duration_ms = events[-1]["time_ms"] if events else 0.0
        tempo_events = sum(1 for e in events if e.get("type") == "tempo")
        log.info(
            "Parsed %d events (duration=%.0f ms, tempo events=%d)", len(events), duration_ms, tempo_events
        )
        return events

    def load_timeline(self, events: List[Dict]) -> MidiPlaybackState:
        """Create playback state from a parsed timeline (immutable event list)."""
        log.debug("Loading timeline with %d events", len(events))
        duration_ms = events[-1]["time_ms"] if events else 0.0
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
            "Initialized playback state (duration=%.0f ms, tempo events=%d)",
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
            state.playback_id += 1
            state.start_clock = None
            state.status = "playing"
            self._wake_event.set()
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
            state.playback_id += 1
            self._wake_event.set()
            log.info("Playback paused at %.0f ms", state.paused_elapsed_ms)

    def resume(self, state: MidiPlaybackState) -> None:
        """Resume playback from paused position."""
        with self._lock:
            if state.status != "paused":
                log.debug("Resume ignored; status=%s", state.status)
                return
            state.start_clock = self._clock()
            state.status = "playing"
            state.playback_id += 1
            self._wake_event.set()
            log.info("Playback resumed at %.0f ms", state.paused_elapsed_ms)

    def stop(self, state: MidiPlaybackState) -> None:
        """Stop playback and reset to start."""
        with self._lock:
            state.status = "stopped"
            state.start_clock = None
            state.paused_elapsed_ms = 0.0
            state.cursor_idx = 0
            state.playback_id += 1
            self._stop_event.set()
            self._wake_event.set()
            log.info("Playback stopped and reset")
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._stop_event.clear()

    def seek(self, state: MidiPlaybackState, target_ms: float) -> None:
        """Seek to target_ms and update cursor; keeps current play/pause status."""
        target_ms = max(0, min(target_ms, state.duration_ms))
        with self._lock:
            state.cursor_idx = self._find_cursor(state.timeline, target_ms)
            state.paused_elapsed_ms = target_ms
            if state.status == "playing":
                state.start_clock = self._clock()
            else:
                state.start_clock = None
            state.playback_id += 1
            self._wake_event.set()
            log.info("Seeked to %.0f ms (cursor=%d)", target_ms, state.cursor_idx)

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
                    if status == "playing" and state.start_clock is None:
                        state.start_clock = self._clock()
                if status == "paused":
                    self._wake_event.wait(self._poll_interval_ms / 1000.0)
                    self._wake_event.clear()
                    continue
                if status != "playing":
                    break
                with self._lock:
                    playback_id = state.playback_id
                now_ms = self._elapsed_ms(state)
                effective_now = now_ms + self._dispatch_ahead_ms
                due_events: List[Dict] = []
                with self._lock:
                    if state.status != "playing" or state.playback_id != playback_id:
                        continue
                    while (
                        state.cursor_idx < len(state.timeline)
                        and state.timeline[state.cursor_idx]["time_ms"] <= effective_now
                    ):
                        due_events.append(state.timeline[state.cursor_idx])
                        state.cursor_idx += 1
                dispatched = len(due_events)
                if dispatched:
                    for event in due_events:
                        with self._lock:
                            if state.playback_id != playback_id or state.status != "playing":
                                break
                        if self._stop_event.is_set():
                            break
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
                if state.cursor_idx < len(state.timeline):
                    next_time = state.timeline[state.cursor_idx]["time_ms"]
                    wait_ms = next_time - (self._elapsed_ms(state) + self._dispatch_ahead_ms)
                    if wait_ms > 0:
                        if self._spin_wait_ms <= 0 or wait_ms > self._spin_wait_ms:
                            sleep_ms = wait_ms - self._spin_wait_ms if wait_ms > self._spin_wait_ms else wait_ms
                            self._wake_event.wait(sleep_ms / 1000.0)
                            self._wake_event.clear()
                        if self._spin_wait_ms > 0 and wait_ms <= self._spin_wait_ms:
                            end = time.perf_counter() + (wait_ms / 1000.0)
                            while time.perf_counter() < end:
                                if self._stop_event.is_set() or self._wake_event.is_set():
                                    self._wake_event.clear()
                                    break
                                time.sleep(0)
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
        elapsed = self._clock() - state.start_clock
        if elapsed < 0:
            return state.paused_elapsed_ms
        return state.paused_elapsed_ms + elapsed

    @staticmethod
    def _find_cursor(timeline: List[Dict], target_ms: float) -> int:
        """Find the first event index at or after target_ms."""
        for idx, event in enumerate(timeline):
            if event["time_ms"] >= target_ms:
                return idx
        return len(timeline)
