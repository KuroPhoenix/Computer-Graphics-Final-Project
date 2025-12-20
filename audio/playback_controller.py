import logging
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from configs import config
from audio.audio_output import BaseAudioOutput, create_audio_output
from audio.midi_manager import MidiManager, MidiPlaybackState
from visuals.piano_sink import NullPianoSink, PianoEventSink

log = logging.getLogger(__name__)

TrackRef = Union[int, str]


class PlaybackController:
    """High-level playback controller for audio and piano event routing."""

    def __init__(
        self,
        tracks: Optional[List[str]] = None,
        sound_font: Optional[str] = None,
        piano_sink: Optional[PianoEventSink] = None,
    ) -> None:
        self._tracks = list(tracks or [])
        self._current_index = 0 if self._tracks else -1
        self._current_path: Optional[str] = self._tracks[self._current_index] if self._current_index >= 0 else None
        self._sound_font = sound_font or getattr(config, "SOUND_FONT_PATH", "")
        self._piano_sink = piano_sink or NullPianoSink()
        self._audio: Optional[BaseAudioOutput] = None
        self._mm: Optional[MidiManager] = None
        self._state: Optional[MidiPlaybackState] = None
        self._dispatch_ahead_ms = 0.0
        self._paused_notes: Dict[Tuple[int, int], int] = {}
        self._paused_time_ms: Optional[float] = None
        self._velocity_scale = 1.0
        self._cc7_scale = 1.0
        self._cc11_scale = 1.0
        self._profile: Dict[str, object] = {}
        self._profile_summary: Optional[Dict[str, object]] = None
        self._filter_controls = set(getattr(config, "FILTER_CONTROLS", ()))
        self._filter_controls_enabled = bool(getattr(config, "FILTER_CONTROL_CHANGES", False))
        self._lock = threading.RLock()
        self._build_audio()

    @property
    def tracks(self) -> List[str]:
        return list(self._tracks)

    def set_tracks(self, tracks: List[str]) -> None:
        with self._lock:
            previous_path = self._current_path
            self._tracks = list(tracks)
            if self._current_path and self._current_path in self._tracks:
                self._current_index = self._tracks.index(self._current_path)
            elif self._tracks and (not self._state or self._state.status != "playing"):
                self._current_index = 0
                self._current_path = self._tracks[0]
            else:
                if not self._tracks:
                    self._current_path = None
                self._current_index = -1
            if previous_path != self._current_path and self._state and self._state.status != "playing":
                self._state = None

    @property
    def current_path(self) -> Optional[str]:
        return self._current_path

    @property
    def sound_font(self) -> str:
        return self._sound_font

    def load_track(self, target: TrackRef) -> bool:
        """Load a track by index or path, replacing the current playback state."""
        path = self._resolve_track(target)
        if not path:
            return False
        with self._lock:
            self.stop()
            if not self._mm:
                self._build_manager()
            try:
                events = self._mm.parse_midi(path)
            except Exception as exc:
                log.error("Failed to parse MIDI %s: %s", path, exc)
                return False
            self._state = self._mm.load_timeline(events)
            self._profile_summary = self._summarize_events(events)
            self._apply_profile_from_summary(self._profile_summary)
            self._current_path = path
        return True

    def analyze_track(self, target: Optional[TrackRef] = None) -> Optional[Dict[str, object]]:
        path = self.resolve_track_path(target)
        if not path:
            return None
        mm = MidiManager(
            poll_interval_ms=getattr(config, "SCHEDULER_POLL_MS", 5.0),
            dispatch_ahead_ms=0.0,
            spin_wait_ms=0.0,
        )
        try:
            events = mm.parse_midi(path)
        except Exception as exc:
            log.error("Failed to parse MIDI %s: %s", path, exc)
            return None
        summary = self._summarize_events(events)
        summary["path"] = path
        return summary

    def play(self) -> bool:
        """Start playback of the current track."""
        with self._lock:
            if not self._audio:
                self._build_audio()
            if not self._mm:
                self._build_manager()
            if not self._state:
                if self._current_path is None and self._tracks:
                    self._current_index = 0
                    self._current_path = self._tracks[0]
                if self._current_path is None:
                    log.warning("No track loaded.")
                    return False
                if not self.load_track(self._current_path):
                    return False
            if self._state.status == "playing":
                log.debug("Playback already running.")
                return True
            self._audio.start_track(self._current_path)
            self._mm.start_playback(self._state, self._on_event, self._on_end)
            return True

    def pause(self) -> None:
        with self._lock:
            if not self._mm or not self._state:
                return
            if self._state.status != "playing":
                return
            now_ms = self._mm.current_position_ms(self._state)
            self._paused_notes = self._active_notes_at(now_ms)
            self._paused_time_ms = now_ms
            self._mm.pause(self._state)
            if self._audio:
                self._audio.panic()
            self._emit_note_offs(self._paused_notes, now_ms)

    def resume(self) -> None:
        with self._lock:
            if not self._mm or not self._state:
                return
            if self._state.status != "paused":
                return
            self._mm.resume(self._state)
            if self._paused_notes and self._paused_time_ms is not None:
                self._retrigger_notes(self._paused_notes, self._paused_time_ms)
            self._paused_notes = {}
            self._paused_time_ms = None

    def stop(self) -> None:
        with self._lock:
            if not self._mm or not self._state:
                return
            now_ms = self._mm.current_position_ms(self._state)
            active_notes = self._active_notes_at(now_ms)
            self._mm.stop(self._state)
            if self._audio:
                self._audio.panic()
            self._emit_note_offs(active_notes, now_ms)
            self._paused_notes = {}
            self._paused_time_ms = None

    def seek(self, target_ms: float) -> None:
        with self._lock:
            if not self._mm or not self._state:
                return
            current_ms = self._mm.current_position_ms(self._state)
            old_active = self._active_notes_at(current_ms)
            self._emit_note_offs(old_active, current_ms)
            self._mm.seek(self._state, target_ms)
            if self._audio:
                self._audio.panic()
            new_active = self._active_notes_at(target_ms)
            self._retrigger_notes(new_active, target_ms)

    def next_track(self, auto_play: bool = True) -> bool:
        with self._lock:
            if not self._tracks:
                return False
            self._current_index = (self._current_index + 1) % len(self._tracks)
            self._current_path = self._tracks[self._current_index]
        if not self.load_track(self._current_path):
            return False
        if auto_play:
            return self.play()
        return True

    def prev_track(self, auto_play: bool = True) -> bool:
        with self._lock:
            if not self._tracks:
                return False
            self._current_index = (self._current_index - 1) % len(self._tracks)
            self._current_path = self._tracks[self._current_index]
        if not self.load_track(self._current_path):
            return False
        if auto_play:
            return self.play()
        return True

    def set_soundfont(self, path: str) -> bool:
        sf_path = Path(path)
        if not sf_path.exists():
            log.error("Soundfont not found: %s", path)
            return False
        with self._lock:
            self.stop()
            self._sound_font = str(sf_path)
            self._build_audio()
            if self._current_path:
                self.load_track(self._current_path)
        return True

    def status(self) -> Dict[str, object]:
        with self._lock:
            state = self._state
            mm = self._mm
            audio = self._audio
            position = mm.current_position_ms(state) if mm and state else 0.0
            return {
                "track": self._current_path,
                "index": self._current_index,
                "status": state.status if state else "idle",
                "pos_ms": position,
                "sound_font": self._sound_font,
                "audio": audio.status() if audio else {},
                "profile": dict(self._profile),
            }

    def shutdown(self) -> None:
        with self._lock:
            if self._mm and self._state:
                self._mm.stop(self._state)
            if self._audio:
                self._audio.panic()
                self._audio.close()
            if self._piano_sink:
                self._piano_sink.close()

    def _build_audio(self) -> None:
        if self._audio:
            try:
                self._audio.close()
            except Exception:
                pass
        self._audio = create_audio_output(sound_font_override=self._sound_font)
        self._dispatch_ahead_ms = self._audio.latency_ms() + getattr(config, "AUDIO_LATENCY_MS", 0.0)
        self._build_manager()
        log.info(
            "Audio backend: %s (latency=%.1f ms, dispatch_ahead=%.1f ms)",
            self._audio.__class__.__name__,
            self._audio.latency_ms(),
            self._dispatch_ahead_ms,
        )

    def _build_manager(self) -> None:
        self._mm = MidiManager(
            poll_interval_ms=getattr(config, "SCHEDULER_POLL_MS", 5.0),
            dispatch_ahead_ms=self._dispatch_ahead_ms,
            spin_wait_ms=getattr(config, "SCHEDULER_SPIN_MS", 0.5),
        )

    def _resolve_track(self, target: TrackRef) -> Optional[str]:
        if isinstance(target, int):
            if target < 0 or target >= len(self._tracks):
                log.error("Track index out of range: %s", target)
                return None
            self._current_index = target
            return self._tracks[target]
        if isinstance(target, str):
            if target.isdigit():
                return self._resolve_track(int(target))
            path = Path(target)
            if path.exists():
                path_str = str(path)
                if path_str in self._tracks:
                    self._current_index = self._tracks.index(path_str)
                return str(path)
            if target in self._tracks:
                self._current_index = self._tracks.index(target)
                return target
            log.error("Track not found: %s", target)
        return None

    def resolve_track_path(self, target: Optional[TrackRef]) -> Optional[str]:
        if target is None:
            return self._current_path
        if isinstance(target, int):
            if 0 <= target < len(self._tracks):
                return self._tracks[target]
            return None
        if isinstance(target, str):
            if target.isdigit():
                return self.resolve_track_path(int(target))
            path = Path(target)
            if path.exists():
                return str(path)
            matches = [
                t
                for t in self._tracks
                if Path(t).name.lower() == target.lower() or Path(t).stem.lower() == target.lower()
            ]
            if len(matches) == 1:
                return matches[0]
        return None

    def _on_event(self, event: dict) -> None:
        if (
            self._filter_controls_enabled
            and event.get("type") == "control_change"
            and event.get("control") in self._filter_controls
        ):
            return
        audio = self._audio
        piano = self._piano_sink
        state = self._state
        mm = self._mm
        profiled = self._apply_event_profile(event)
        if audio:
            audio.handle_event(profiled)
        if profiled.get("type") in ("on", "off") and piano:
            playback_time = mm.current_position_ms(state) if mm and state else None
            piano.handle_note_event(profiled, playback_time_ms=playback_time)

    def _on_end(self) -> None:
        if self._audio:
            self._audio.panic()
        log.info("Track finished: %s", self._current_path)

    def _apply_profile_from_summary(self, summary: Dict[str, object]) -> None:
        self._profile = {}
        self._velocity_scale = 1.0
        self._cc7_scale = 1.0
        self._cc11_scale = 1.0

        if not getattr(config, "AUTO_PROFILE", False):
            return

        avg_velocity = float(summary.get("velocity_avg", 0.0) or 0.0)
        if avg_velocity > 0:
            target = float(getattr(config, "TARGET_AVG_VELOCITY", avg_velocity))
            scale = target / avg_velocity
            min_scale = float(getattr(config, "VELOCITY_SCALE_MIN", 0.7))
            max_scale = float(getattr(config, "VELOCITY_SCALE_MAX", 1.3))
            self._velocity_scale = max(min(scale, max_scale), min_scale)
        self._profile["velocity_scale"] = self._velocity_scale

        if getattr(config, "AUTO_CC7_SCALE", False):
            cc7_avg = float(summary.get("cc7_avg", 0.0) or 0.0)
            if cc7_avg > 0:
                target = float(getattr(config, "TARGET_CC7", cc7_avg))
                scale = target / cc7_avg
                min_scale = float(getattr(config, "CC7_SCALE_MIN", 0.8))
                max_scale = float(getattr(config, "CC7_SCALE_MAX", 1.2))
                self._cc7_scale = max(min(scale, max_scale), min_scale)
            self._profile["cc7_scale"] = self._cc7_scale

        if getattr(config, "AUTO_CC11_SCALE", False):
            cc11_avg = float(summary.get("cc11_avg", 0.0) or 0.0)
            if cc11_avg > 0:
                target = float(getattr(config, "TARGET_CC11", cc11_avg))
                scale = target / cc11_avg
                min_scale = float(getattr(config, "CC11_SCALE_MIN", 0.8))
                max_scale = float(getattr(config, "CC11_SCALE_MAX", 1.2))
                self._cc11_scale = max(min(scale, max_scale), min_scale)
            self._profile["cc11_scale"] = self._cc11_scale

        if getattr(config, "AUTO_GAIN", False):
            base_gain = float(getattr(config, "FLUID_GAIN", 0.8))
            poly_target = float(getattr(config, "POLYPHONY_TARGET", 8.0))
            max_poly = float(summary.get("max_polyphony", 0.0) or 0.0)
            if max_poly > 0:
                poly_factor = (poly_target / max_poly) ** 0.5
            else:
                poly_factor = 1.0
            gain = base_gain * poly_factor
            gain_min = float(getattr(config, "GAIN_MIN", 0.2))
            gain_max = float(getattr(config, "GAIN_MAX", 0.8))
            gain = max(min(gain, gain_max), gain_min)
            self._profile["gain"] = gain
            if self._audio:
                try:
                    self._audio.apply_profile(self._profile)
                except Exception:
                    pass

    def _scaled_velocity(self, velocity: int) -> int:
        scaled = int(round(velocity * self._velocity_scale))
        return max(1, min(127, scaled))

    def _apply_event_profile(self, event: dict) -> dict:
        etype = event.get("type")
        if etype == "on":
            if self._velocity_scale == 1.0:
                return event
            updated = dict(event)
            updated["velocity"] = self._scaled_velocity(event.get("velocity", 0))
            return updated
        if etype == "control_change":
            ctrl = event.get("control", 0)
            scale = 1.0
            if ctrl == 7:
                scale = self._cc7_scale
            elif ctrl == 11:
                scale = self._cc11_scale
            if scale != 1.0:
                updated = dict(event)
                val = int(round(updated.get("value", 0) * scale))
                updated["value"] = max(0, min(127, val))
                return updated
        return event

    @staticmethod
    def _summarize_events(events: List[Dict]) -> Dict[str, object]:
        notes = 0
        channels = Counter()
        programs: Dict[int, set] = defaultdict(set)
        control_counts = Counter()
        cc7_values: List[int] = []
        cc11_values: List[int] = []
        pitch_bends = 0
        aftertouch = 0
        poly_aftertouch = 0
        velocity_sum = 0
        velocity_min = None
        velocity_max = None
        note_min = None
        note_max = None
        active: Dict[Tuple[int, int], int] = {}
        max_polyphony = 0
        duration_ms = events[-1]["time_ms"] if events else 0.0

        for event in events:
            etype = event.get("type")
            if etype in ("on", "off"):
                ch = event.get("channel", 0)
                channels[ch] += 1
                key = (ch, event.get("note", 0))
                if etype == "on" and event.get("velocity", 0) > 0:
                    notes += 1
                    velocity = event.get("velocity", 0)
                    active[key] = velocity
                    velocity_sum += velocity
                    velocity_min = velocity if velocity_min is None else min(velocity_min, velocity)
                    velocity_max = velocity if velocity_max is None else max(velocity_max, velocity)
                    note = event.get("note", 0)
                    note_min = note if note_min is None else min(note_min, note)
                    note_max = note if note_max is None else max(note_max, note)
                else:
                    active.pop(key, None)
                max_polyphony = max(max_polyphony, len(active))
            elif etype == "program_change":
                ch = event.get("channel", 0)
                programs[ch].add(event.get("program", 0))
            elif etype == "control_change":
                ctrl = event.get("control", 0)
                control_counts[ctrl] += 1
                val = event.get("value", 0)
                if ctrl == 7:
                    cc7_values.append(val)
                elif ctrl == 11:
                    cc11_values.append(val)
            elif etype == "pitchwheel":
                pitch_bends += 1
            elif etype == "channel_pressure":
                aftertouch += 1
            elif etype == "poly_aftertouch":
                poly_aftertouch += 1

        def _avg(values: List[int]) -> float:
            return sum(values) / len(values) if values else 0.0

        return {
            "duration_ms": duration_ms,
            "notes": notes,
            "channels": sorted(channels.keys()),
            "channel_note_counts": dict(channels),
            "program_changes": {ch: sorted(list(vals)) for ch, vals in programs.items()},
            "control_changes": dict(control_counts),
            "cc7_min": min(cc7_values) if cc7_values else 0,
            "cc7_max": max(cc7_values) if cc7_values else 0,
            "cc7_avg": _avg(cc7_values),
            "cc11_min": min(cc11_values) if cc11_values else 0,
            "cc11_max": max(cc11_values) if cc11_values else 0,
            "cc11_avg": _avg(cc11_values),
            "pitch_bends": pitch_bends,
            "aftertouch": aftertouch,
            "poly_aftertouch": poly_aftertouch,
            "max_polyphony": max_polyphony,
            "velocity_min": velocity_min or 0,
            "velocity_max": velocity_max or 0,
            "velocity_avg": (velocity_sum / notes) if notes else 0.0,
            "note_min": note_min or 0,
            "note_max": note_max or 0,
        }

    def _active_notes_at(self, time_ms: float) -> Dict[Tuple[int, int], int]:
        if not self._state:
            return {}
        active: Dict[Tuple[int, int], int] = {}
        for event in self._state.timeline:
            if event["time_ms"] > time_ms:
                break
            if event.get("type") == "on":
                key = (event.get("channel", 0), event["note"])
                active[key] = event.get("velocity", 64)
            elif event.get("type") == "off":
                key = (event.get("channel", 0), event["note"])
                active.pop(key, None)
        return active

    def _retrigger_notes(self, notes: Dict[Tuple[int, int], int], time_ms: float) -> None:
        for (channel, note), velocity in notes.items():
            event = {
                "type": "on",
                "note": note,
                "velocity": self._scaled_velocity(velocity),
                "channel": channel,
                "time_ms": time_ms,
            }
            if self._audio:
                self._audio.handle_event(event)
            if self._piano_sink:
                self._piano_sink.handle_note_event(event, playback_time_ms=time_ms)

    def _emit_note_offs(self, notes: Dict[Tuple[int, int], int], time_ms: float) -> None:
        for (channel, note) in notes.keys():
            event = {
                "type": "off",
                "note": note,
                "velocity": 0,
                "channel": channel,
                "time_ms": time_ms,
            }
            if self._piano_sink:
                self._piano_sink.handle_note_event(event, playback_time_ms=time_ms)
