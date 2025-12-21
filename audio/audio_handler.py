from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterable, Optional

from configs import config
from audio.playback_controller import PlaybackController
from visuals.piano_sink import CompositePianoSink, NoteCaptureSink, NullPianoSink, PianoEventSink


class AudioHandler:
    """Single entry point for song discovery, playback, and note reporting."""

    def __init__(
        self,
        midi_root: Optional[str] = None,
        scan_recursive: Optional[bool] = None,
        extensions: Optional[Iterable[str]] = None,
        sound_font: Optional[str] = None,
        piano_sink: Optional[PianoEventSink] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._midi_root = Path(midi_root or getattr(config, "MIDI_SCAN_ROOT", "midi"))
        self._scan_recursive = bool(
            getattr(config, "MIDI_SCAN_RECURSIVE", True) if scan_recursive is None else scan_recursive
        )
        ext_cfg = extensions or getattr(config, "MIDI_EXTENSIONS", (".mid", ".midi"))
        self._extensions = tuple(ext.lower() for ext in ext_cfg)

        self._note_capture = NoteCaptureSink()
        self._piano_sink = piano_sink or NullPianoSink()
        self._composite_sink = CompositePianoSink([self._piano_sink, self._note_capture])

        self._controller = PlaybackController(
            tracks=[],
            sound_font=sound_font or getattr(config, "SOUND_FONT_PATH", ""),
            piano_sink=self._composite_sink,
        )

        self._library: list[dict] = []
        self._active_folder_index = -1
        self._current_track_index = -1
        self._current_path: Optional[str] = None

        self.Rescan()

    @property
    def active_folder_index(self) -> int:
        return self._active_folder_index

    @property
    def current_track_index(self) -> int:
        return self._current_track_index

    @property
    def current_path(self) -> Optional[str]:
        return self._current_path

    @property
    def sound_font(self) -> str:
        return self._controller.sound_font

    def List_Songs(self) -> list[dict]:
        with self._lock:
            listing = []
            for folder_index, entry in enumerate(self._library):
                tracks = [
                    {"track_index": idx, "name": Path(path).name, "path": path}
                    for idx, path in enumerate(entry["tracks"])
                ]
                listing.append(
                    {
                        "folder_index": folder_index,
                        "label": entry["label"],
                        "path": entry["path"],
                        "tracks": tracks,
                    }
                )
            return listing

    def Play_Song(self, selection: object = None) -> bool:
        with self._lock:
            if not self._library:
                return False

            if selection is not None:
                resolved = self._resolve_selection(selection)
                if not resolved:
                    return False
                folder_index, track_index, path = resolved
                if folder_index is not None and folder_index != self._active_folder_index:
                    if not self._set_active_folder(folder_index):
                        return False
                if track_index is not None:
                    if not self._controller.load_track(track_index):
                        return False
                    self._current_track_index = track_index
                    self._current_path = self._controller.current_path
                elif path:
                    if not self._controller.load_track(path):
                        return False
                    self._current_path = self._controller.current_path
                    self._current_track_index = self._index_in_active_folder(self._current_path)
            elif self._current_path is None:
                tracks = self._active_tracks()
                if not tracks:
                    return False
                if not self._controller.load_track(0):
                    return False
                self._current_track_index = 0
                self._current_path = self._controller.current_path

            if self._controller.play():
                self._current_path = self._controller.current_path
                self._current_track_index = self._index_in_active_folder(self._current_path)
                return True
            return False

    def Pause_Song(self) -> None:
        self._controller.pause()

    def Resume_Song(self) -> None:
        self._controller.resume()

    def Stop_Song(self) -> None:
        self._controller.stop()

    def Next_song(self) -> bool:
        with self._lock:
            if self._controller.next_track(auto_play=True):
                self._current_path = self._controller.current_path
                self._current_track_index = self._index_in_active_folder(self._current_path)
                return True
            return False

    def Prev_song(self) -> bool:
        with self._lock:
            if self._controller.prev_track(auto_play=True):
                self._current_path = self._controller.current_path
                self._current_track_index = self._index_in_active_folder(self._current_path)
                return True
            return False

    def Seek(self, target_ms: float) -> None:
        self._controller.seek(target_ms)

    def Report_Note(self, drain: bool = False) -> Optional[dict] | list[dict]:
        if drain:
            return self._note_capture.drain()
        event, playback_time_ms, active_notes = self._note_capture.snapshot()
        if event is None:
            return None
        report = dict(event)
        if playback_time_ms is not None:
            report["playback_time_ms"] = playback_time_ms
        if active_notes:
            report["active_notes"] = active_notes
        return report

    def Rescan(self) -> list[dict]:
        with self._lock:
            previous_path = self._current_path
            previous_folder_path = None
            if 0 <= self._active_folder_index < len(self._library):
                previous_folder_path = self._library[self._active_folder_index]["path"]

            self._library = self._build_library()
            if not self._library:
                self._active_folder_index = -1
                self._current_track_index = -1
                self._current_path = None
                self._controller.set_tracks([])
                return []

            folder_index = None
            track_index = None
            if previous_path:
                folder_index, track_index = self._find_track_by_path(previous_path)
            if folder_index is None and previous_folder_path:
                for idx, entry in enumerate(self._library):
                    if entry["path"] == previous_folder_path:
                        folder_index = idx
                        break
            if folder_index is None:
                folder_index = 0

            self._active_folder_index = folder_index
            tracks = self._library[self._active_folder_index]["tracks"]
            self._controller.set_tracks(tracks)

            if track_index is not None and 0 <= track_index < len(tracks):
                self._current_track_index = track_index
                self._current_path = tracks[track_index]
            elif tracks:
                self._current_track_index = 0
                self._current_path = tracks[0]
            else:
                self._current_track_index = -1
                self._current_path = None

            return self.List_Songs()

    def Set_Folder(self, folder_index: int) -> bool:
        with self._lock:
            return self._set_active_folder(folder_index)

    def Analyze_Song(self, selection: object = None) -> Optional[dict]:
        with self._lock:
            target_path = None
            if selection is None:
                target_path = self._current_path
            else:
                resolved = self._resolve_selection(selection)
                if not resolved:
                    return None
                _, track_index, path = resolved
                if track_index is not None:
                    tracks = self._active_tracks()
                    if 0 <= track_index < len(tracks):
                        target_path = tracks[track_index]
                else:
                    target_path = path
            if not target_path:
                return None
        return self._controller.analyze_track(target_path)

    def Status(self) -> dict:
        return self._controller.status()

    def List_Soundfonts(self) -> list[Path]:
        font_dir = Path("midi/soundfonts")
        if not font_dir.exists():
            return []
        return sorted(font_dir.glob("*.sf2"))

    def Set_Soundfont(self, selection: object) -> bool:
        path = None
        if isinstance(selection, int):
            fonts = self.List_Soundfonts()
            if 0 <= selection < len(fonts):
                path = str(fonts[selection])
        elif isinstance(selection, str):
            candidate = Path(selection)
            if candidate.exists():
                path = str(candidate)
            else:
                for font in self.List_Soundfonts():
                    if selection.lower() in (font.name.lower(), font.stem.lower()):
                        path = str(font)
                        break
        if not path:
            return False
        return self._controller.set_soundfont(path)

    def Shutdown(self) -> None:
        self._controller.shutdown()
        self._composite_sink.close()

    def _scan_midi_paths(self, root: Optional[Path]) -> list[Path]:
        if root and root.exists():
            iterator = root.rglob("*") if self._scan_recursive else root.glob("*")
            paths = [
                path for path in iterator if path.is_file() and path.suffix.lower() in self._extensions
            ]
            paths.sort(key=lambda p: (str(p.parent).lower(), p.name.lower()))
            return paths
        return [Path(p) for p in getattr(config, "MIDI_FILES", []) if Path(p).exists()]

    def _build_library(self) -> list[dict]:
        root = self._midi_root if self._midi_root.exists() else None
        paths = self._scan_midi_paths(root)
        if not paths:
            return []

        groups: dict[str, dict] = {}
        for path in paths:
            folder = path.parent
            if root:
                try:
                    rel = folder.relative_to(root)
                    label = str(rel) if str(rel) != "." else root.name
                except ValueError:
                    label = str(folder)
            else:
                label = str(folder)
            if label not in groups:
                groups[label] = {"label": label, "path": str(folder), "tracks": []}
            groups[label]["tracks"].append(str(path))

        library = [groups[key] for key in sorted(groups.keys(), key=lambda k: k.lower())]
        for entry in library:
            entry["tracks"].sort(key=lambda p: Path(p).name.lower())
        return library

    def _active_tracks(self) -> list[str]:
        if 0 <= self._active_folder_index < len(self._library):
            return list(self._library[self._active_folder_index]["tracks"])
        return []

    def _find_track_by_path(self, path: str) -> tuple[Optional[int], Optional[int]]:
        for folder_index, entry in enumerate(self._library):
            if path in entry["tracks"]:
                return folder_index, entry["tracks"].index(path)
        return None, None

    def _index_in_active_folder(self, path: Optional[str]) -> int:
        if not path:
            return -1
        tracks = self._active_tracks()
        if path in tracks:
            return tracks.index(path)
        return -1

    def _resolve_selection(self, selection: object) -> Optional[tuple[Optional[int], Optional[int], Optional[str]]]:
        folder_index = None
        track_index = None
        path = None

        if isinstance(selection, dict):
            if "path" in selection:
                path = selection.get("path")
            if "folder_index" in selection:
                folder_index = selection.get("folder_index")
            if "track_index" in selection:
                track_index = selection.get("track_index")
        elif isinstance(selection, (tuple, list)) and len(selection) == 2:
            folder_index, track_index = selection
        elif isinstance(selection, int):
            track_index = selection
        elif isinstance(selection, str):
            if ":" in selection:
                parts = selection.split(":", 1)
                if parts[0].isdigit() and parts[1].isdigit():
                    folder_index = int(parts[0])
                    track_index = int(parts[1])
            if path is None:
                candidate = Path(selection)
                if candidate.exists():
                    path = str(candidate)
                else:
                    match = self._match_by_name(selection, self._active_folder_index)
                    if match is None:
                        match = self._match_by_name(selection, None)
                    if match is not None:
                        folder_index, track_index, path = match

        if folder_index is not None:
            if not isinstance(folder_index, int) or not (0 <= folder_index < len(self._library)):
                return None
        if track_index is not None:
            if not isinstance(track_index, int):
                return None
            if folder_index is None:
                folder_index = self._active_folder_index
            tracks = self._library[folder_index]["tracks"] if folder_index >= 0 else []
            if not (0 <= track_index < len(tracks)):
                return None
            path = tracks[track_index]
        elif path:
            folder_match, track_match = self._find_track_by_path(path)
            if folder_match is not None:
                folder_index = folder_match if folder_index is None else folder_index
                track_index = track_match

        if folder_index is not None and track_index is None and path is None:
            tracks = self._library[folder_index]["tracks"]
            if tracks:
                track_index = 0
                path = tracks[0]

        if folder_index is None and path is None:
            return None
        return folder_index, track_index, path

    def _match_by_name(
        self, name: str, folder_index: Optional[int]
    ) -> Optional[tuple[int, int, str]]:
        candidates = []
        if folder_index is None:
            folders = enumerate(self._library)
        else:
            if 0 <= folder_index < len(self._library):
                folders = [(folder_index, self._library[folder_index])]
            else:
                return None
        for fidx, entry in folders:
            for tidx, path in enumerate(entry["tracks"]):
                stem = Path(path).stem.lower()
                filename = Path(path).name.lower()
                if name.lower() in (stem, filename):
                    candidates.append((fidx, tidx, path))
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _set_active_folder(self, folder_index: int) -> bool:
        if not (0 <= folder_index < len(self._library)):
            return False
        status = self._controller.status().get("status")
        if status in ("playing", "paused"):
            self._controller.stop()

        self._active_folder_index = folder_index
        tracks = self._library[folder_index]["tracks"]
        self._controller.set_tracks(tracks)

        if self._current_path in tracks:
            self._current_track_index = tracks.index(self._current_path)
        elif tracks:
            self._current_track_index = 0
            self._current_path = tracks[0]
        else:
            self._current_track_index = -1
            self._current_path = None
        return True
