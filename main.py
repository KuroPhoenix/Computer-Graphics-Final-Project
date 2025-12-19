import time

from configs.logging_config import get_logger
from audio.midi_manager import MidiManager
from audio.audio_output import create_audio_output
from configs import config

log = get_logger(__name__)


def run_track(path: str) -> None:
    log.info("Starting track: %s", path)
    mm = MidiManager()
    events = mm.parse_midi(path)
    state = mm.load_timeline(events)
    audio = create_audio_output()
    log.info("Using audio backend: %s", audio.__class__.__name__)

    def on_event(ev):
        audio.handle_event(ev)

    def on_end():
        log.info("Track finished: %s", path)
        audio.panic()
        audio.close()

    try:
        audio.start_track(path)
        mm.start_playback(state, on_event, on_end)
        # Block until playback completes or is stopped
        last_log = time.perf_counter()
        while state.status != "stopped":
            time.sleep(0.05)
            now = time.perf_counter()
            if now - last_log >= 1.0:
                pos_ms = mm.current_position_ms(state)
                audio_status = getattr(audio, "status", lambda: {} )()
                log.debug(
                    "Monitor: pos_ms=%.0f, cursor=%d/%d, status=%s, audio=%s",
                    pos_ms,
                    state.cursor_idx,
                    len(state.timeline),
                    state.status,
                    audio_status,
                )
                last_log = now
    except KeyboardInterrupt:
        log.info("Interrupted; stopping playback.")
        mm.stop(state)
        audio.panic()
    finally:
        audio.close()


if __name__ == "__main__":
    for midi_path in config.MIDI_FILES:
        run_track(midi_path)
