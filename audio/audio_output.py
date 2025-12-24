import logging
import os
from pathlib import Path
from typing import Optional

import pygame
import pygame.midi

from configs import config

log = logging.getLogger(__name__)


class BaseAudioOutput:
    """Interface for audio backends."""

    def start_track(self, path: str) -> None:
        """Called once per track before event dispatch."""
        return

    def handle_event(self, event: dict) -> None:
        """Handle a parsed event (note on/off)."""
        return

    def panic(self) -> None:
        """Stop all sound."""
        return

    def pause(self) -> None:
        """Pause playback when supported; defaults to panic."""
        self.panic()

    def resume(self) -> None:
        """Resume playback when supported."""
        return

    def close(self) -> None:
        """Release resources."""
        return

    def status(self) -> dict:
        """Return backend-specific status for monitoring."""
        return {"backend": self.__class__.__name__}

    def latency_ms(self) -> float:
        """Estimated output latency in milliseconds."""
        return 0.0

    def apply_profile(self, profile: dict) -> None:
        """Apply per-track playback profile (optional)."""
        return


class MidiAudioOutput(BaseAudioOutput):
    """
    Note-level MIDI output using pygame.midi.
    Requires a MIDI synth; on WSL, configure FluidSynth/Timidity or a virtual device.
    """

    def __init__(self, device_id: Optional[int] = None):
        pygame.midi.init()
        self._device_id = device_id if device_id is not None else pygame.midi.get_default_output_id()
        if self._device_id == -1:
            raise RuntimeError("No default MIDI output device found. Configure a synth backend.")
        self._output = pygame.midi.Output(self._device_id)
        self._closed = False
        self._latency_ms = float(getattr(config, "MIDI_LATENCY_MS", 0.0))
        log.info("MidiAudioOutput initialized (device_id=%s)", self._device_id)

    def handle_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "on":
            self._start_note(event["note"], event.get("velocity", 64), event.get("channel", 0))
        elif etype == "off":
            self._stop_note(event["note"], event.get("channel", 0))
        elif etype == "program_change":
            self._program_change(event.get("program", 0), event.get("channel", 0))
        elif etype == "control_change":
            self._control_change(event.get("control", 0), event.get("value", 0), event.get("channel", 0))
        elif etype == "pitchwheel":
            self._pitch_bend(event.get("pitch", 0), event.get("channel", 0))
        elif etype == "channel_pressure":
            self._channel_pressure(event.get("pressure", 0), event.get("channel", 0))
        elif etype == "poly_aftertouch":
            self._poly_aftertouch(
                event.get("note", 0),
                event.get("pressure", 0),
                event.get("channel", 0),
            )

    def _start_note(self, note: int, velocity: int, channel: int = 0) -> None:
        if self._closed:
            return
        try:
            self._output.note_on(note, velocity, channel)
        except Exception as exc:
            log.error("Failed to start note %s: %s", note, exc)

    def _stop_note(self, note: int, channel: int = 0) -> None:
        if self._closed:
            return
        try:
            self._output.note_off(note, 0, channel)
        except Exception as exc:
            log.error("Failed to stop note %s: %s", note, exc)

    def _program_change(self, program: int, channel: int = 0) -> None:
        if self._closed:
            return
        try:
            self._output.set_instrument(program, channel)
        except Exception as exc:
            log.error("Failed to send program change %s: %s", program, exc)

    def _control_change(self, control: int, value: int, channel: int = 0) -> None:
        if self._closed:
            return
        try:
            self._output.write_short(0xB0 + channel, control & 0x7F, value & 0x7F)
        except Exception as exc:
            log.error("Failed to send control change %s=%s: %s", control, value, exc)

    def _pitch_bend(self, pitch: int, channel: int = 0) -> None:
        if self._closed:
            return
        bend = max(0, min(16383, pitch + 8192))
        lsb = bend & 0x7F
        msb = (bend >> 7) & 0x7F
        try:
            self._output.write_short(0xE0 + channel, lsb, msb)
        except Exception as exc:
            log.error("Failed to send pitch bend %s: %s", pitch, exc)

    def _channel_pressure(self, pressure: int, channel: int = 0) -> None:
        if self._closed:
            return
        try:
            self._output.write_short(0xD0 + channel, pressure & 0x7F, 0)
        except Exception as exc:
            log.error("Failed to send channel pressure %s: %s", pressure, exc)

    def _poly_aftertouch(self, note: int, pressure: int, channel: int = 0) -> None:
        if self._closed:
            return
        try:
            self._output.write_short(0xA0 + channel, note & 0x7F, pressure & 0x7F)
        except Exception as exc:
            log.error("Failed to send poly aftertouch note=%s: %s", note, exc)

    def panic(self) -> None:
        if self._closed:
            return
        for ch in range(16):
            try:
                self._output.write_short(0xB0 + ch, 123, 0)  # controller 123: all notes off
            except Exception as exc:
                log.error("Failed to send all-notes-off on channel %d: %s", ch, exc)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._output.close()
        finally:
            self._closed = True
            pygame.midi.quit()

    def status(self) -> dict:
        return {
            "backend": "MidiAudioOutput",
            "closed": self._closed,
            "device_id": self._device_id,
            "latency_ms": self._latency_ms,
        }

    def latency_ms(self) -> float:
        return self._latency_ms


class PyFluidSynthOutput(BaseAudioOutput):
    """
    In-process FluidSynth backend (per-note) using pyfluidsynth; does not require ALSA seq.
    """

    def __init__(self, sound_font: str, gain: float = 0.8):
        dll_hint = getattr(config, "FLUIDSYNTH_DLL_PATH", "") or ""
        driver_cfg = getattr(config, "FLUID_DRIVER", "auto").lower()
        # Allow adding an explicit DLL search path before importing pyfluidsynth (Windows)
        if dll_hint:
            dll_path = Path(dll_hint)
            if dll_path.exists():
                try:
                    os.add_dll_directory(str(dll_path))
                    os.environ["PATH"] = f"{dll_path};{os.environ.get('PATH','')}"
                    log.info("Added FluidSynth DLL path: %s", dll_path)
                except Exception as exc:  # pragma: no cover - platform-specific
                    log.warning("Failed to add DLL path %s: %s", dll_path, exc)
        try:
            import fluidsynth  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "pyfluidsynth import failed; install pyfluidsynth and ensure FluidSynth DLLs are discoverable "
                "(set config.FLUIDSYNTH_DLL_PATH to your FluidSynth \\bin folder on Windows)."
            ) from exc
        if getattr(config, "FLUID_SUPPRESS_LOGS", False):
            self._suppress_fluidsynth_logs(fluidsynth)

        self._fs = fluidsynth.Synth()
        self._gain = gain
        # Choose driver candidates
        if driver_cfg == "auto":
            if os.name == "nt":
                driver_candidates = ["wasapi", "dsound", "sdl2", "portaudio"]
            else:
                driver_candidates = ["pulseaudio", "alsa", "jack", "portaudio"]
        else:
            driver_candidates = [driver_cfg]
        sample_rate = getattr(config, "FLUID_SAMPLE_RATE", 44100.0)
        period_size = getattr(config, "FLUID_PERIOD_SIZE", 256)
        periods = getattr(config, "FLUID_PERIODS", 4)
        interp = getattr(config, "FLUID_INTERP", None)
        reverb = getattr(config, "FLUID_REVERB", None)
        chorus = getattr(config, "FLUID_CHORUS", None)
        polyphony = getattr(config, "FLUID_POLYPHONY", None)
        latency_override = getattr(config, "FLUID_LATENCY_MS", None)
        if latency_override is None:
            self._latency_ms = (period_size * periods / float(sample_rate)) * 1000.0
        else:
            self._latency_ms = float(latency_override)
        pulse_latency = getattr(config, "PULSE_LATENCY_MSEC", None)
        opts = {
            "synth.sample-rate": sample_rate,
            "audio.period-size": period_size,
            "audio.periods": periods,
            "synth.gain": gain,
        }
        midi_driver = getattr(config, "FLUID_MIDI_DRIVER", None)
        if midi_driver:
            opts["midi.driver"] = midi_driver
        if interp is not None:
            opts["synth.interpolation"] = int(interp)
        if reverb is not None:
            opts["synth.reverb.active"] = int(bool(reverb))
        if chorus is not None:
            opts["synth.chorus.active"] = int(bool(chorus))
        if polyphony is not None:
            opts["synth.polyphony"] = int(polyphony)
        for key, val in opts.items():
            try:
                self._fs.setting(key, val)
            except Exception as exc:
                log.debug("PyFluidSynthOutput could not set %s=%s (%s)", key, val, exc)

        started = False
        for driver in driver_candidates:
            try:
                if driver == "pulseaudio" and pulse_latency is not None and pulse_latency > 0:
                    os.environ.setdefault("PULSE_LATENCY_MSEC", str(int(pulse_latency)))
                self._fs.setting("audio.driver", driver)
                self._fs.start(driver=driver)
                self._driver = driver
                started = True
                log.info(
                    "PyFluidSynthOutput using driver=%s sample_rate=%.1f period_size=%s periods=%s gain=%.2f",
                    driver,
                    sample_rate,
                    period_size,
                    periods,
                    gain,
                )
                break
            except Exception as exc:
                log.warning("PyFluidSynthOutput driver '%s' failed (%s)", driver, exc)
        if not started:
            self._fs.start()
            self._driver = "default"
            log.info("PyFluidSynthOutput using driver=default")

        try:
            self._sfid = self._fs.sfload(sound_font)
        except Exception as exc:
            log.error("Failed to load soundfont %s: %s", sound_font, exc)
            raise

        self._force_program = getattr(config, "FORCE_PROGRAM", False)
        self._force_bank = getattr(config, "FORCE_PROGRAM_BANK", 0)
        self._force_preset = getattr(config, "FORCE_PROGRAM_PRESET", 0)
        self._bank_msb = [0] * 16
        self._bank_lsb = [0] * 16
        self._invalid_presets = set()
        # Set a default program on all channels; optionally force preset for consistency.
        if self._force_program:
            bank_msb = (self._force_bank >> 7) & 0x7F
            bank_lsb = self._force_bank & 0x7F
            for ch in range(16):
                self._bank_msb[ch] = bank_msb
                self._bank_lsb[ch] = bank_lsb
                if not self._program_select(ch, self._force_bank, self._force_preset, context="force"):
                    self._program_select(ch, 0, 0, context="force-fallback")
        else:
            enable_drums = bool(getattr(config, "ENABLE_GM_DRUMS", True))
            for ch in range(16):
                bank = 128 if enable_drums and ch == 9 else 0
                self._bank_msb[ch] = (bank >> 7) & 0x7F
                self._bank_lsb[ch] = bank & 0x7F
                if not self._program_select(ch, bank, 0, context="init") and bank != 0:
                    self._bank_msb[ch] = 0
                    self._bank_lsb[ch] = 0
                    self._program_select(ch, 0, 0, context="init-fallback")

        self._events_handled = 0
        self._closed = False
        log.info("PyFluidSynthOutput initialized with %s", sound_font)

    def handle_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "on":
            self._fs.noteon(event.get("channel", 0), event["note"], event.get("velocity", 64))
            self._events_handled += 1
        elif etype == "off":
            self._fs.noteoff(event.get("channel", 0), event["note"])
            self._events_handled += 1
        elif etype == "control_change":
            ch = event.get("channel", 0)
            ctrl = event.get("control", 0)
            val = event.get("value", 0)
            val = max(0, min(127, val))
            if ctrl == 0:
                self._bank_msb[ch] = val
            elif ctrl == 32:
                self._bank_lsb[ch] = val
            self._fs.cc(ch, ctrl, val)
            self._events_handled += 1
        elif etype == "pitchwheel":
            bend = event.get("pitch", 0) + 8192
            bend = max(0, min(16383, bend))
            self._fs.pitch_bend(event.get("channel", 0), bend)
            self._events_handled += 1
        elif etype == "channel_pressure":
            if hasattr(self._fs, "channel_pressure"):
                self._fs.channel_pressure(event.get("channel", 0), event.get("pressure", 0))
            self._events_handled += 1
        elif etype == "poly_aftertouch":
            # PyFluidSynth doesn't expose poly aftertouch; ignore gracefully.
            self._events_handled += 1
        elif etype == "program_change":
            if not self._force_program:
                program = event.get("program", 0)
                ch = event.get("channel", 0)
                bank = (self._bank_msb[ch] << 7) + self._bank_lsb[ch]
                if not self._program_select(ch, bank, program, context="event"):
                    if bank != 0 or program != 0:
                        self._program_select(ch, 0, 0, context="event-fallback")
            self._events_handled += 1

    @staticmethod
    def _suppress_fluidsynth_logs(fluidsynth_module) -> None:
        def _noop(*_args, **_kwargs) -> None:
            return

        levels = []
        for name in ("FLUID_PANIC", "FLUID_ERR", "FLUID_WARN", "FLUID_INFO", "FLUID_DBG"):
            level = getattr(fluidsynth_module, name, None)
            if isinstance(level, int):
                levels.append(level)
        if not levels:
            levels = [0, 1, 2, 3, 4]

        for fn_name in ("set_log_function", "fluid_set_log_function"):
            fn = getattr(fluidsynth_module, fn_name, None)
            if not callable(fn):
                continue
            for level in levels:
                try:
                    fn(level, _noop)
                except TypeError:
                    try:
                        fn(level, _noop, None)
                    except Exception:
                        continue
                except Exception:
                    continue

    def _program_select(self, channel: int, bank: int, preset: int, context: str = "") -> bool:
        try:
            result = self._fs.program_select(channel, self._sfid, bank, preset)
        except Exception as exc:
            key = (bank, preset)
            if key not in self._invalid_presets:
                log.warning(
                    "Program select failed (bank=%d preset=%d ch=%d ctx=%s): %s",
                    bank,
                    preset,
                    channel,
                    context,
                    exc,
                )
                self._invalid_presets.add(key)
            return False
        if result not in (None, 0):
            key = (bank, preset)
            if key not in self._invalid_presets:
                log.warning(
                    "Program select rejected (bank=%d preset=%d ch=%d ctx=%s)",
                    bank,
                    preset,
                    channel,
                    context,
                )
                self._invalid_presets.add(key)
            return False
        return True

    def panic(self) -> None:
        if self._closed:
            return
        self._fs.system_reset()

    def pause(self) -> None:
        if self._closed:
            return
        for ch in range(16):
            try:
                self._fs.cc(ch, 120, 0)  # all sound off
                self._fs.cc(ch, 123, 0)  # all notes off
            except Exception:
                continue

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._fs.delete()
        finally:
            self._closed = True

    def status(self) -> dict:
        return {
            "backend": "PyFluidSynthOutput",
            "closed": self._closed,
            "driver": getattr(self, "_driver", "unknown"),
            "latency_ms": self._latency_ms,
            "gain": getattr(self, "_gain", None),
        }

    def latency_ms(self) -> float:
        return self._latency_ms

    def apply_profile(self, profile: dict) -> None:
        gain = profile.get("gain")
        if gain is None:
            return
        self._gain = float(gain)
        if self._closed:
            return
        try:
            if hasattr(self._fs, "set_gain"):
                self._fs.set_gain(self._gain)
            else:
                self._fs.setting("synth.gain", self._gain)
        except Exception:
            return


class MixerAudioOutput(BaseAudioOutput):
    """
    File-level playback using pygame.mixer. Audio timing is handled by SDL, not per-event notes.
    Useful when no MIDI device is available.
    """

    def __init__(self):
        self._started = False
        self._closed = False
        try:
            pygame.mixer.init()
            self._init_ok = True
            log.info("MixerAudioOutput initialized")
        except Exception as exc:
            self._init_ok = False
            log.error("Mixer init failed: %s", exc)

    def start_track(self, path: str) -> None:
        if not getattr(self, "_init_ok", False) or self._closed:
            return
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            self._started = True
            log.info("Mixer playback started for %s", path)
        except Exception as exc:
            log.error("Failed to play %s via mixer: %s", path, exc)

    def handle_event(self, event: dict) -> None:
        # Mixer backend does not handle per-note events; audio is handled by mixer.music
        return

    def pause(self) -> None:
        if not getattr(self, "_init_ok", False) or self._closed:
            return
        if self._started:
            pygame.mixer.music.pause()

    def resume(self) -> None:
        if not getattr(self, "_init_ok", False) or self._closed:
            return
        if self._started:
            pygame.mixer.music.unpause()

    def panic(self) -> None:
        if not getattr(self, "_init_ok", False) or self._closed:
            return
        if self._started:
            pygame.mixer.music.stop()

    def close(self) -> None:
        if self._closed or not getattr(self, "_init_ok", False):
            return
        try:
            if self._started:
                pygame.mixer.music.stop()
        finally:
            self._closed = True
            pygame.mixer.quit()

    def status(self) -> dict:
        busy = False
        pos_ms = None
        if getattr(self, "_init_ok", False) and not self._closed:
            try:
                busy = pygame.mixer.music.get_busy()
                pos_ms = pygame.mixer.music.get_pos()
            except Exception:
                busy = False
        return {
            "backend": "MixerAudioOutput",
            "closed": self._closed,
            "init_ok": getattr(self, "_init_ok", False),
            "busy": busy,
            "pos_ms": pos_ms,
        }

    def latency_ms(self) -> float:
        return 0.0


class NoAudioOutput(BaseAudioOutput):
    """No-op backend when audio is disabled."""

    def __init__(self):
        log.warning("No audio backend configured; running silent.")


def create_audio_output(sound_font_override: Optional[str] = None) -> BaseAudioOutput:
    """Factory to create the configured audio backend."""
    backend = getattr(config, "AUDIO_BACKEND", "midi").lower()
    device_id = getattr(config, "AUDIO_MIDI_DEVICE_ID", None)
    sf_path = sound_font_override or getattr(config, "SOUND_FONT_PATH", "")
    gain = getattr(config, "FLUID_GAIN", 0.8)

    if backend == "pyfluidsynth":
        try:
            ao = PyFluidSynthOutput(sound_font=sf_path, gain=gain)
            log.info("Audio backend: pyfluidsynth")
            return ao
        except Exception as exc:
            log.error("pyfluidsynth backend failed (%s); falling back to mixer.", exc)
            backend = "mixer"

    if backend == "midi":
        try:
            ao = MidiAudioOutput(device_id=device_id)
            log.info("Audio backend: pygame.midi (device_id=%s)", ao._device_id)
            return ao
        except Exception as exc:
            log.error("MIDI backend failed (%s); falling back to mixer.", exc)
            backend = "mixer"

    if backend == "mixer":
        try:
            ao = MixerAudioOutput()
            log.info("Audio backend: mixer")
            return ao
        except Exception as exc:
            log.error("Mixer backend failed (%s); running without audio.", exc)
            return NoAudioOutput()

    log.warning("Audio backend set to 'none' or unknown; running silent.")
    return NoAudioOutput()
