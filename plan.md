---
name: plan
description: AudioHandler single entry point and piano rendering integration plan
---

# Plan

Build an `AudioHandler` class as the sole entry point for playback and note dispatch, using the current MIDI scheduling/audio stack as an internal dependency. This plan folds in the current codebase state (CLI scanning, playback controller, note routing) and adds a GUI-ready API surface, a robust song library index, and a piano-rendering event pipeline. The approach is to keep the timing engine unchanged and wrap it behind a stable, thread-safe handler interface.

## Project scan summary
- `main.py` already implements folder-indexed scanning and a CLI loop, but it talks directly to `PlaybackController` and contains scanning logic that should move into `AudioHandler` for GUI reuse.
- `audio/playback_controller.py` already routes note on/off events to a piano sink via `_on_event`, making it the best place to attach a capture sink for `Report_Note()`.
- `audio/midi_manager.py` parses MIDI into a time-ordered event list and runs a scheduler thread with latency compensation; this is the timing core the handler should keep intact.
- `audio/audio_output.py` is the audio backend switchboard (pyfluidsynth/pygame.midi/mixer), with per-track gain control already supported via `apply_profile`.
- `visuals/piano_sink.py` defines a `PianoEventSink` interface but no concrete renderer; this is where GUI rendering hooks will plug in.
- `configs/config.py` holds scan defaults and playback tuning; the handler should honor these values.

## Requirements
- Provide a single class `AudioHandler` with the exact API:
  - `List_Songs()`: list available songs (indexed, folder-aware).
  - `Play_Song()`: play the current (or selected) song.
  - `Pause_Song()`: pause playback.
  - `Resume_Song()`: resume playback.
  - `Next_song()`: play next song in the current folder.
  - `Report_Note()`: return current MIDI note info (spec fields, plus timing).
- AudioHandler is the only point of entry for GUI and any other module; no other module touches `PlaybackController` directly.
- Note events must be dispatched from the handler to a piano renderer sink.
- Song scanning must be automatic and folder-indexed, matching current CLI behavior.
- Preserve the existing timing and auto-analysis behavior in `PlaybackController`.

## Scope
- In:
  - New `audio/audio_handler.py` that wraps `PlaybackController` and exposes the GUI-safe API.
  - Move or reuse the current scan/indexing logic from `main.py` into `AudioHandler`.
  - Add a note capture sink (thread-safe) to support `Report_Note()`.
  - Attach a piano renderer sink via the handler (fan-out dispatch).
  - Refactor `main.py` to use only `AudioHandler` (no direct controller usage).
  - Update `README.md` to document handler usage and the piano rendering pipeline.
- Out:
  - GUI windowing/toolkit code.
  - DSP changes or soundfont policy changes.

## Files and entry points
- `audio/audio_handler.py`: new entry class used by CLI and GUI.
- `audio/playback_controller.py`: remains internal; no external callers after refactor.
- `visuals/piano_sink.py`: interface for renderer sink; extend with concrete sink(s).
- `main.py`: refactor to be a thin CLI wrapper around `AudioHandler`.
- `README.md`: document handler interface and rendering plan.

## API sketch (signatures only)

```python
class AudioHandler:
    def List_Songs(self) -> list[dict]: ...
    def Play_Song(self, selection: str | int | None = None) -> bool: ...
    def Pause_Song(self) -> None: ...
    def Resume_Song(self) -> None: ...
    def Next_song(self) -> bool: ...
    def Report_Note(self) -> dict | None: ...
```

## Data model / API changes
- `List_Songs()` returns a folder-grouped list with stable indices, for GUI binding:
  - `{folder_index, label, path, tracks:[{track_index, name, path}]}`
- `Report_Note()` returns a dict mirroring MIDI event fields:
  - `{"type","note","velocity","channel","time_ms","playback_time_ms"}`
  - Optional `"active_notes"` list if the GUI needs chord state.

## Implementation notes (AudioHandler responsibilities)
- Own the scan/index logic currently in `main.py` (`_scan_midi_paths`, `_build_library`), but keep it internal so GUI never re-implements scanning.
- Own a `PlaybackController` instance; `Play_Song()` must call `load_track()` before `play()` to ensure auto-analysis is applied.
- Provide a method or constructor parameter to attach a piano sink (renderer). The handler should be the place to wire a `CompositeSink` so both the renderer and a `NoteCaptureSink` receive events.
- Provide `Report_Note()` as a thread-safe read of the last note event and/or active note list.

## 3D piano rendering plan (GUI-facing)
- Use a render thread that owns visual state; it consumes note events from the handler sink.
- Use a thread-safe queue for events; renderer drains all events <= current render time to avoid lag.
- Use a Collada (.dae) piano model for the static body/skeleton; keys are animated independently.
- Best case: the DAE has each key as a separate node/mesh; animate by rotating or translating the key node around its hinge.
- Fallback: if keys are merged, treat the DAE as a static shell and render instanced key meshes separately.
- Map MIDI notes to 88 keys (A0=21 to C8=108) and bind each note to a key node or instanced key transform.
- Visual response:
  - `note_on`: rotate the key around its hinge and modulate material brightness by velocity.
  - `note_off`: return to rest with a short easing (e.g., 30–60 ms).
- Timing:
  - Use `playback_time_ms` from the handler to align animation with audio.
  - If the GUI frame drops, drain queued events in a tight loop to catch up.

Example animation pseudocode:
```python
def on_note_event(event):
    key = key_map[event["note"]]
    if event["type"] == "on":
        key.target_angle = key.rest_angle - key.press_angle
        key.intensity = event["velocity"] / 127.0
    else:
        key.target_angle = key.rest_angle

def render_frame(dt):
    for key in keys:
        key.angle = lerp(key.angle, key.target_angle, key.ease)
        set_node_transform(key.node, rotate_x(key.angle))
        set_material_intensity(key.node, key.intensity)
```

## Action items
[ ] Create `audio/audio_handler.py` and define the public API, returning deterministic structures for the GUI.
[ ] Move the scan/index logic from `main.py` into the handler and expose `List_Songs()`.
[ ] Implement `NoteCaptureSink` + `CompositePianoSink` to support both renderer dispatch and `Report_Note()`.
[ ] Update `PlaybackController` usage to be internal-only and update `main.py` to call `AudioHandler`.
[ ] Add README updates covering handler usage and the piano rendering pipeline.

## Testing and validation
- Manual CLI sanity:
  - Instantiate `AudioHandler`, call `List_Songs()`, and verify stable indices.
  - Call `Play_Song()`, `Pause_Song()`, `Resume_Song()`, `Next_song()` and verify state transitions.
  - Poll `Report_Note()` during playback and verify note-on/off data is updated.
- Threading check:
  - Simulate rapid play/pause/seek while polling `Report_Note()` to confirm no race exceptions.

## Risks and edge cases
- Index stability if rescan occurs while the GUI has a selection.
- Event queue growth if GUI render thread stalls (needs backpressure or max queue size).
- `Report_Note()` returning stale data when a track is paused or idle.
- Duplicate song names across folders: GUI should rely on indices not names.

## Open questions
- Should we include PEP-8 aliases (`list_songs`, `play_song`, etc.) in addition to exact method names?
- Should `Report_Note()` return last event only, or last event plus currently active notes?
- Should `List_Songs()` return a flat list and a grouped list to simplify GUI binding?
- Does the DAE model include each key as a separate node/mesh, or only a merged keybed?
