# Computer-Graphics-Final-Project
NYCU Computer Graphics (Steve Lin) Final Project

This project is a MIDI player and visualization pipeline. It parses MIDI files into a single
time-ordered event stream, schedules playback with sub-millisecond precision, and routes events
to both audio output and optional visual handlers (virtual piano, FX, etc).

## Quick start
1) `python -m main`
2) Use the CLI menu to select and play tracks.

The CLI auto-scans the `midi/` folder for `.mid`/`.midi` files and presents indexed menus.

## CLI usage
Key commands:
- `menu` / `list`: show folders + track indexes
- `folders`: list folders (indexed)
- `tracks [folder_index]`: list tracks in a folder
- `use <folder_index>`: set active folder
- `play <index>` or `play <folder:index>`: play a track
- `pause`, `resume`, `stop`, `next`, `prev`, `seek <ms>`
- `soundfonts`, `soundfont <index|path>`
- `analyze <index|folder:index>`: show per-track stats

## Architecture
Core modules:
- `audio/midi_manager.py`: loads MIDI, builds the event timeline, and runs the scheduler.
- `audio/playback_controller.py`: high-level control (load/play/pause/seek) and event routing.
- `audio/audio_output.py`: audio backends (FluidSynth, pygame.midi, mixer).
- `visuals/piano_sink.py`: interface for a virtual piano or other visual consumers.
- `main.py`: CLI wrapper around `PlaybackController`.

## Event schema
Events are plain dicts. Common fields:
```
{
  "time_ms": float,
  "type": "on" | "off" | "control_change" | "program_change" | "pitchwheel" | ...,
  "note": int,          # for note events
  "velocity": int,      # for note-on
  "channel": int,
  "control": int,       # for control_change
  "value": int          # for control_change
}
```

`PlaybackController` forwards note on/off events to the piano sink and forwards all MIDI events
to the audio backend (subject to normalization rules).

## Handlers (for future development)

### Audio handlers
Implement a new backend by subclassing `BaseAudioOutput` in `audio/audio_output.py`:
```
class MyAudioOutput(BaseAudioOutput):
    def start_track(self, path: str) -> None: ...
    def handle_event(self, event: dict) -> None: ...
    def panic(self) -> None: ...
    def close(self) -> None: ...
    def latency_ms(self) -> float: ...
```
Register it in `create_audio_output()` and select it via `config.AUDIO_BACKEND`.

### Piano/visual handlers
Implement `PianoEventSink` in `visuals/piano_sink.py`:
```
class MyPianoSink(PianoEventSink):
    def handle_note_event(self, event: dict, playback_time_ms: float | None = None) -> None: ...
    def close(self) -> None: ...
```
Attach it by constructing `PlaybackController(piano_sink=MyPianoSink())`
or by updating `main.py` to use your sink.

### FX handlers
If you need separate FX routing, follow the same pattern as `PianoEventSink`:
- Create a new interface (e.g., `FxEventSink`)
- Route it from `PlaybackController._on_event()`
- Keep it event-driven to stay in sync with audio

## Tuning / normalization
Per-track analysis runs automatically before playback (`AUTO_PROFILE = True`). Useful knobs in
`configs/config.py`:
- `TARGET_AVG_VELOCITY`, `VELOCITY_SCALE_MIN/MAX`
- `AUTO_CC7_SCALE`, `AUTO_CC11_SCALE` (volume/expression normalization)
- `AUTO_GAIN` (caps gain based on polyphony)
- `FILTER_CONTROL_CHANGES` + `FILTER_CONTROLS` to ignore tone-shaping CCs

These settings help reduce perceived differences across MIDI files when using a single
piano soundfont.
