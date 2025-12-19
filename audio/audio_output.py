import logging
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

    def close(self) -> None:
        """Release resources."""
        return

    def status(self) -> dict:
        """Return backend-specific status for monitoring."""
        return {"backend": self.__class__.__name__}


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
        log.info("MidiAudioOutput initialized (device_id=%s)", self._device_id)

    def handle_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "on":
            self._start_note(event["note"], event.get("velocity", 64), event.get("channel", 0))
        elif etype == "off":
            self._stop_note(event["note"], event.get("channel", 0))

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
        return {"backend": "MidiAudioOutput", "closed": self._closed, "device_id": self._device_id}


class PyFluidSynthOutput(BaseAudioOutput):
    """
    In-process FluidSynth backend (per-note) using pyfluidsynth; does not require ALSA seq.
    """

    def __init__(self, sound_font: str, gain: float = 0.8):
        try:
            import fluidsynth  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyfluidsynth not installed; pip install pyfluidsynth") from exc

        self._fs = fluidsynth.Synth()
        # Try to configure audio settings even without Settings API.
        driver = "pulseaudio"
        opts = {
            "audio.driver": driver,
            "synth.sample-rate": 44100.0,
            "audio.period-size": 256,
            "audio.periods": 4,
            "synth.gain": gain,
        }
        for key, val in opts.items():
            try:
                self._fs.setting(key, val)
            except Exception as exc:
                log.debug("PyFluidSynthOutput could not set %s=%s (%s)", key, val, exc)

        try:
            self._fs.start(driver=driver)
            log.info(
                "PyFluidSynthOutput using driver=%s sample_rate=%.1f period_size=%s periods=%s gain=%.2f",
                driver,
                opts["synth.sample-rate"],
                opts["audio.period-size"],
                opts["audio.periods"],
                gain,
            )
            self._driver = driver
        except Exception as exc:
            log.warning("PyFluidSynthOutput pulseaudio driver failed (%s); trying portaudio", exc)
            try:
                driver = "portaudio"
                self._fs.setting("audio.driver", driver)
                self._fs.start(driver=driver)
                self._driver = driver
                log.info("PyFluidSynthOutput using driver=%s (fallback)", driver)
            except Exception as exc2:
                log.warning("PyFluidSynthOutput portaudio driver failed (%s); using default", exc2)
                self._fs.start()
                self._driver = "default"

        try:
            self._sfid = self._fs.sfload(sound_font)
        except Exception as exc:
            log.error("Failed to load soundfont %s: %s", sound_font, exc)
            raise
        # Set a default program on all channels; program changes will override per channel.
        for ch in range(16):
            self._fs.program_select(ch, self._sfid, 0, 0)

        self._closed = False
        log.info("PyFluidSynthOutput initialized with %s", sound_font)

    def handle_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "on":
            self._fs.noteon(event.get("channel", 0), event["note"], event.get("velocity", 64))
        elif etype == "off":
            self._fs.noteoff(event.get("channel", 0), event["note"])
        elif etype == "program_change":
            program = event.get("program", 0)
            ch = event.get("channel", 0)
            self._fs.program_select(ch, self._sfid, 0, program)

    def panic(self) -> None:
        if self._closed:
            return
        self._fs.system_reset()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._fs.delete()
        finally:
            self._closed = True

    def status(self) -> dict:
        return {"backend": "PyFluidSynthOutput", "closed": self._closed, "driver": getattr(self, "_driver", "unknown")}


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


class NoAudioOutput(BaseAudioOutput):
    """No-op backend when audio is disabled."""

    def __init__(self):
        log.warning("No audio backend configured; running silent.")


def create_audio_output() -> BaseAudioOutput:
    """Factory to create the configured audio backend."""
    backend = getattr(config, "AUDIO_BACKEND", "midi").lower()
    device_id = getattr(config, "AUDIO_MIDI_DEVICE_ID", None)
    sf_path = getattr(config, "SOUND_FONT_PATH", "")
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
