# Computer-Graphics-Final-Project
NYCU Computer Graphics (Steve Lin) Final Project

My plan is to implement a midi player and music visualiser. In particular, I want to render a piano that plays midi files, with some special effects rendered. 
I am thinking to split my code into 3 parts, midi manager, piano renderer and special effect generator. For midi manager, its primary function is to parse midi files, 
play midi music and send notes to the piano to render pressing that note. As of now, I plan to have midi manager return on a per-note basis, and the piano renderer simply takes in 
that note and render the piano key being pressed. However, I am also contemplating on sending the entire parsed MIDI dict to piano due to latency concerns, I am conerned that 
there might be latencies where the render rate cannot keep up with the music. And the note should also be passed to special effects generator, where it will do stuff (dont know yet) based on the music it receives. 

Right now, my focus is, where should the client code be, or rather, how the client code could look like. I want the client to see the piano, 
and can freely resume, pause and choose which music to import and have the piano play.

## Playback Pipeline (proposed)

```
MIDI file -> parse_midi -> event timeline -> scheduler -> {audio, piano renderer, FX renderer}
```

Single source of truth is the time-stamped event timeline. Everything (audio trigger, piano visuals, special effects) listens to the same stream to avoid drift.

### Data Model
- `NoteEvent`: `{time_ms, type: "on"|"off", note: int, velocity: int, channel: int}`
- `Timeline`: ordered list of `NoteEvent` plus tempo/time signature changes if needed.

### MIDI Manager
- `parse_midi(file_path) -> Timeline`  
  Input: path to `.mid`. Output: sorted `Timeline`. Process: load MIDI, extract note-on/off with absolute times (ms), include tempo changes.
- `load_timeline(timeline) -> PlaybackState`  
  Input: `Timeline`. Output: state with cursor, tempo map. Process: reset playback position to start.
- `schedule_playback(state, on_event, on_end, clock)`  
  Input: `PlaybackState`, callbacks, clock function returning current ms. Output: runs scheduler loop/thread. Process: while playing, dispatch events whose `time_ms <= clock()` to `on_event(event)`, stop at end and call `on_end()`. Supports pause/resume by freezing base clock offset.
- `set_play_state(state, mode)`  
  Input: state, mode (`play|pause|stop|seek`). Output: updates internal offsets. Process: adjust start time and cursor; on `seek`, move cursor to first event >= target time.

### Audio Output
- `play_audio_event(event)`  
  Input: `NoteEvent`. Output: sends to MIDI synth or mixer. Process: note-on/off mapped to audio backend (e.g., pygame.midi or fluidsynth).
- `stop_audio()`  
  Stops all sounding notes (used on stop/seek).

### Piano Renderer (visual)
- `queue_event(event)`  
  Input: `NoteEvent`. Output: adds to thread-safe queue for render thread.
- `render_frame(dt, event_queue)`  
  Input: delta time, queue. Output: draws piano keys. Process: drain events whose `time_ms` <= now, update key states; if frames drop, fast-forward by consuming all due events.

### Special Effects Generator
- `consume_event(event)`  
  Input: `NoteEvent`. Output: updates FX triggers (e.g., particles, spectrum). Process: map note/velocity to effect parameters.
- `render_fx(dt)`  
  Input: delta time. Output: draws FX layer using last triggered states.

### Client / UI Layer
- `load_track(path)` -> calls `parse_midi`, `load_timeline`.
- `play() / pause() / resume() / stop() / seek(ms)` -> delegate to scheduler helpers.
- `select_track_from_ui(path)` -> calls `load_track`, then `play`.

### Timing Notes
- Keep audio and visuals decoupled: scheduler fires events to both; audio backend handles sound, render loop handles frames.
- Use one clock source (`time.perf_counter` in Python) passed to scheduler to avoid mixing clocks.
- On pause, record elapsed playback time; on resume, shift the base start time so event times stay aligned.



cat > ~/.asoundrc <<'EOF'                                                                                                                                                                              
  pcm.!default {                                                                                                                                                                                         
      type pulse                                                                                                                                                                                         
  }                                                                                                                                                                                                      
  ctl.!default {                                                                                                                                                                                         
      type pulse                                                                                                                                                                                         
  }                                                                                                                                                                                                      
  EOF  