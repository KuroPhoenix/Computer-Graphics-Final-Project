import atexit
import os
import sys
import threading
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

from audio.audio_handler import AudioHandler
from visuals import piano_on_wave


def _find_fontaine_selection(handler: AudioHandler) -> object | None:
    for folder in handler.List_Songs():
        for track in folder["tracks"]:
            if "fontaine" in track["name"].lower():
                return {"folder_index": folder["folder_index"], "track_index": track["track_index"]}
    return None


def _pump_notes(handler: AudioHandler, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        events = handler.Report_Note(drain=True)
        if events:
            for event in events:
                piano_on_wave.send_midi_event(event)
        time.sleep(0.005)


def main() -> int:
    handler = AudioHandler()
    selection = _find_fontaine_selection(handler)
    if selection is None:
        selection = 0

    stop_event = threading.Event()
    pump_thread: threading.Thread | None = None
    ui_state = {"playing": False}

    def _track_label() -> str:
        path = handler.current_path
        if path:
            return Path(path).name
        return "No track"

    def _refresh_ui(playing: bool | None = None) -> None:
        if playing is not None:
            ui_state["playing"] = playing
        piano_on_wave.set_ui_state(
            playing=ui_state["playing"],
            track_name=_track_label(),
        )

    def _toggle_play_pause() -> None:
        status = handler.Status().get("status")
        if status == "playing":
            handler.Pause_Song()
            _refresh_ui(False)
        elif status == "paused":
            handler.Resume_Song()
            _refresh_ui(True)
        else:
            if handler.Play_Song():
                _refresh_ui(True)
            else:
                _refresh_ui(False)

    def _next_track() -> None:
        if handler.Next_song():
            _refresh_ui(True)
        else:
            _refresh_ui(ui_state["playing"])

    def _start_playback() -> None:
        nonlocal pump_thread
        if not handler.Play_Song(selection):
            print("Unable to start playback.")
            _refresh_ui(False)
            return
        pump_thread = threading.Thread(target=_pump_notes, args=(handler, stop_event), daemon=True)
        pump_thread.start()
        _refresh_ui(True)

    def _shutdown() -> None:
        stop_event.set()
        handler.Shutdown()

    atexit.register(_shutdown)

    _refresh_ui(False)
    piano_on_wave.set_ui_callbacks(play_pause=_toggle_play_pause, next_track=_next_track)
    piano_on_wave.run(debug=False, on_ready=_start_playback)
    return 0


if __name__ == "__main__":
    sys.exit(main())
