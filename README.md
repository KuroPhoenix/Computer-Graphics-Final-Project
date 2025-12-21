# Computer-Graphics-Final-Project
NYCU Computer Graphics (Steve Lin) Final Project

This project is a MIDI player and visualization pipeline. It parses MIDI files into a single
time-ordered event stream, schedules playback with sub-millisecond precision, and routes events
to both audio output and optional visual handlers (virtual piano, FX, etc).

## Quick start
Visualization demo (auto-plays Fontaine):
```bash
python -m main
```

CLI player:
```bash
python -m audio_cli
```

The CLI auto-scans the `midi/` folder for `.mid`/`.midi` files and presents indexed menus.

## Setup
Prerequisites:
- Python 3.11+ (tested with 3.13)
- Optional: FluidSynth installed (recommended for `pyfluidsynth` backend)

RECOMMENDED: Python environment (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
Note: If executed under WSL environment, desyncing/jittering issues may occur. 

Python environment (macOS/Linux):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

FluidSynth (for `pyfluidsynth` backend):
- Install FluidSynth and make sure its DLLs/shared libs are discoverable.
- If needed, set `FLUIDSYNTH_DLL_PATH` in `configs/config.py` to the FluidSynth `bin` folder.
- Ensure a soundfont is available (defaults to `midi/soundfonts/salamander.sf2`).

Run the visual demo:
```bash
python -m main
```

Run the CLI:
```bash
python -m audio_cli
```

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

## AudioHandler (single entry point)
`AudioHandler` (implemented in `audio/audio_handler.py`) is the single entry point for GUI
and CLI playback operations. Other modules should not touch `PlaybackController` directly.
This isolates timing, scanning, backend configuration, and note dispatch behind a single API.

### What AudioHandler owns
- **Song library:** scans `midi/` (and subfolders) and returns a stable, indexed menu.
- **Selection state:** tracks the active folder and current song index.
- **Playback orchestration:** `load_track()` + `play()` internally, preserving auto-analysis.
- **Event routing:** forwards note on/off to the piano renderer sink.
- **Note reporting:** provides `Report_Note()` for UI overlays or debug panels.
- **Thread safety:** hides scheduler thread details from the GUI.

### Internal composition (data flow)
```
MIDI file -> MidiManager -> event timeline -> PlaybackController -> Audio backend
                                                  |
                                                  +-> PianoEventSink (renderer)
                                                  +-> NoteCaptureSink (Report_Note)
```

### API (single entry point)
```python
from audio.audio_handler import AudioHandler

handler = AudioHandler(piano_sink=MyPianoSink())

songs = handler.List_Songs()
handler.Play_Song()
handler.Pause_Song()
handler.Resume_Song()
handler.Next_song()

note = handler.Report_Note()
```

### List_Songs() return shape (GUI menu)
Stable indices are required for UI selection widgets.
```python
[
  {
    "folder_index": 0,
    "label": "tracks",
    "path": "midi/tracks",
    "tracks": [
      {"track_index": 0, "name": "natlan.mid", "path": "midi/tracks/natlan.mid"},
      {"track_index": 1, "name": "Inazuma.mid", "path": "midi/tracks/Inazuma.mid"},
    ],
  }
]
```

### Selecting and playing a song
`Play_Song()` should accept an optional selection to update the current song before playback.
```python
songs = handler.List_Songs()
handler.Play_Song(selection={"folder_index": 0, "track_index": 1})
```
If no selection is provided, the handler plays the current song (or the first in the folder).

### Report_Note() return shape
`Report_Note()` returns the latest note event (or `None` if idle). Use
`Report_Note(drain=True)` to retrieve all pending note events since the last call.
```python
{
  "type": "on",
  "note": 60,
  "velocity": 96,
  "channel": 0,
  "time_ms": 1234.0,
  "playback_time_ms": 1238.4
}
```

```python
[
  {"type": "on", "note": 60, "velocity": 96, "channel": 0, "time_ms": 1234.0},
  {"type": "off", "note": 60, "velocity": 0, "channel": 0, "time_ms": 1370.0},
]
```

time_ms is the MIDI timeline timestamp of the event (absolute time since track start, set during parsing). 

playback_time_ms is the actual playback clock position when the event is dispatched to the sink, so it reflects pauses/seeks and real-time scheduling (and can drift slightly due to dispatch timing). 

See audio/midi_manager.py for time_ms creation and audio/playback_controller.py for playback_time_ms capture.                                                                                                                                                                              

• Example: the MIDI event is scheduled at 1000 ms from the start of the song, but you paused for 250 ms before it fired.
                                                                                                                                                                                                         
  Timeline                                                                                                                                                                                               
                                                                                                                                                                                                         
  - T=0: play                                                                                                                                                                                            
  - T=800: pause for 250 ms                                                                                                                                                                              
  - T=1050: resume                                                                                                                                                                                       
  - Event originally at time_ms=1000 fires at playback_time_ms=1250

### Piano renderer integration
The renderer should consume note events from a thread-safe queue so it never blocks
the scheduler thread.
```python
class MyPianoSink(PianoEventSink):
    def __init__(self):
        self.queue = collections.deque()
        self.lock = threading.Lock()

    def handle_note_event(self, event, playback_time_ms=None):
        with self.lock:
            self.queue.append((event, playback_time_ms))

    def pop_events(self):
        with self.lock:
            events = list(self.queue)
            self.queue.clear()
        return events
```

```python
# GUI render loop
for event, t in piano_sink.pop_events():
    if event["type"] == "on":
        press_key(event["note"], event["velocity"])
    else:
        release_key(event["note"])
```

### Threading and timing notes
- The scheduler runs in a background thread; GUI code must never block it.
- `Report_Note()` is thread-safe; prefer `Report_Note(drain=True)` when driving rendering.
- Renderers should drain event queues every frame to stay in sync with audio.
