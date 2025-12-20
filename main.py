import os
import shlex
import sys
import warnings
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

from configs import config

if getattr(config, "CLI_SUPPRESS_STDERR", False):
    stderr_path = Path("logs/cli_stderr.log")
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_file = stderr_path.open("a", buffering=1)
    sys.stderr = stderr_file
    try:
        os.dup2(stderr_file.fileno(), 2)
    except Exception:
        pass

from audio.playback_controller import PlaybackController
from visuals.piano_sink import NullPianoSink

SOUNDFONT_DIR = Path("midi/soundfonts")
MIDI_ROOT = Path(getattr(config, "MIDI_SCAN_ROOT", "midi"))
MIDI_SCAN_RECURSIVE = bool(getattr(config, "MIDI_SCAN_RECURSIVE", True))
MIDI_EXTENSIONS = tuple(
    ext.lower() for ext in getattr(config, "MIDI_EXTENSIONS", (".mid", ".midi"))
)


def _print_help() -> None:
    print(
        "Commands:\n"
        "  menu | list                   Show folders and track indexes\n"
        "  folders                       Show indexed folders\n"
        "  tracks [folder_index]         Show tracks for a folder\n"
        "  use <folder_index>            Set the active folder\n"
        "  play [index|folder:index]     Play a track (current if omitted)\n"
        "  analyze [index|folder:index]  Show MIDI channel/program stats\n"
        "  rescan                        Rescan MIDI folders\n"
        "  pause                         Pause playback\n"
        "  resume                        Resume playback\n"
        "  stop                          Stop playback\n"
        "  next                          Play next track\n"
        "  prev                          Play previous track\n"
        "  seek <ms>                     Seek to position in ms\n"
        "  soundfonts                    List available soundfonts\n"
        "  soundfont [index|path]        Show or set soundfont\n"
        "  status                        Show current status\n"
        "  help                          Show this help\n"
        "  quit                          Exit\n"
        "\n"
        "Shortcuts:\n"
        "  <index>                       Play track by index in active folder\n"
        "  <folder:index>                Play track by folder and index (e.g. 1:3)\n"
        "  <name|path>                   Play track by name or path"
    )


def _scan_midi_paths(root: Path, recursive: bool, extensions: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if path.is_file() and path.suffix.lower() in extensions:
            paths.append(path)
    paths.sort(key=lambda p: (str(p.parent).lower(), p.name.lower()))
    return paths


def _build_library() -> list[dict]:
    root = MIDI_ROOT if MIDI_ROOT.exists() else None
    paths = _scan_midi_paths(root, MIDI_SCAN_RECURSIVE, MIDI_EXTENSIONS) if root else []
    if not paths:
        paths = [Path(p) for p in getattr(config, "MIDI_FILES", []) if Path(p).exists()]
        root = None

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


def _folder_tracks(library: list[dict], folder_index: int) -> list[str]:
    if folder_index < 0 or folder_index >= len(library):
        return []
    return list(library[folder_index]["tracks"])


def _print_folders(library: list[dict], active_folder: int) -> None:
    if not library:
        print("No MIDI folders found.")
        return
    print("Folders (use 'use <index>' to select):")
    for idx, entry in enumerate(library):
        marker = "*" if idx == active_folder else " "
        label = entry["label"]
        count = len(entry["tracks"])
        print(f"{marker} [{idx}] {label} ({count} tracks) -> {entry['path']}")


def _print_tracks_for_folder(library: list[dict], folder_index: int, current_path: str | None) -> None:
    if not library:
        print("No MIDI tracks found.")
        return
    if folder_index < 0 or folder_index >= len(library):
        print("Folder index out of range.")
        return
    entry = library[folder_index]
    print(
        f"Tracks in folder [{folder_index}] {entry['label']} "
        f"(use 'play <index>' or 'play {folder_index}:<index>'):"
    )
    for idx, path in enumerate(entry["tracks"]):
        marker = "*" if path == current_path else " "
        name = Path(path).name
        print(f"{marker} [{idx}] {name} -> {path}")


def _print_menu(library: list[dict], active_folder: int, current_path: str | None) -> None:
    _print_folders(library, active_folder)
    if library:
        _print_tracks_for_folder(library, active_folder, current_path)


def _list_soundfonts() -> list[Path]:
    if not SOUNDFONT_DIR.exists():
        return []
    return sorted(SOUNDFONT_DIR.glob("*.sf2"))


def _print_soundfonts(controller: PlaybackController) -> None:
    fonts = _list_soundfonts()
    if not fonts:
        print("No soundfonts found.")
        return
    print("Soundfonts (use 'soundfont <index>' to switch):")
    for idx, path in enumerate(fonts):
        marker = "*" if str(path) == controller.sound_font else " "
        print(f"{marker} [{idx}] {path.name} -> {path}")


def _print_status(controller: PlaybackController) -> None:
    status = controller.status()
    print(f"track: {status.get('track')}")
    print(f"index: {status.get('index')}")
    print(f"state: {status.get('status')}")
    print(f"pos_ms: {status.get('pos_ms'):.0f}")
    print(f"soundfont: {status.get('sound_font')}")
    print(f"audio: {status.get('audio')}")


def _print_analysis(summary: dict) -> None:
    path = summary.get("path")
    duration = summary.get("duration_ms", 0.0)
    channels = summary.get("channels", [])
    programs = summary.get("program_changes", {})
    control_changes = summary.get("control_changes", {})
    max_polyphony = summary.get("max_polyphony", 0)
    velocity_min = summary.get("velocity_min", 0)
    velocity_max = summary.get("velocity_max", 0)
    velocity_avg = summary.get("velocity_avg", 0.0)
    note_min = summary.get("note_min", 0)
    note_max = summary.get("note_max", 0)
    cc7_avg = summary.get("cc7_avg", 0.0)
    cc11_avg = summary.get("cc11_avg", 0.0)
    print(f"track: {path}")
    print(f"duration_ms: {duration:.0f}")
    print(f"notes: {summary.get('notes', 0)}")
    print(f"channels: {channels}")
    print(f"max_polyphony: {max_polyphony}")
    print(f"velocity: min={velocity_min} max={velocity_max} avg={velocity_avg:.1f}")
    print(f"note_range: {note_min}-{note_max}")
    if cc7_avg:
        print(f"cc7_avg: {cc7_avg:.1f}")
    if cc11_avg:
        print(f"cc11_avg: {cc11_avg:.1f}")
    if programs:
        print(f"program_changes: {programs}")
    if control_changes:
        top_controls = sorted(control_changes.items(), key=lambda kv: kv[1], reverse=True)[:6]
        print(f"control_changes(top): {top_controls}")
        if control_changes.get(64):
            print(f"note: sustain pedal used ({control_changes.get(64)} events).")
        if control_changes.get(11):
            print(f"note: expression (CC11) used ({control_changes.get(11)} events).")
        if control_changes.get(7):
            print(f"note: volume (CC7) used ({control_changes.get(7)} events).")
    if 9 in channels:
        print("note: channel 10 (drums) is used; piano-only soundfonts will remap drums to piano.")
    if getattr(config, "FORCE_PROGRAM", False) and programs:
        print("note: FORCE_PROGRAM is enabled; multi-instrument tracks will be forced to a single preset.")


def _parse_folder_track(token: str, active_folder: int) -> tuple[int, int] | None:
    if ":" in token:
        parts = token.split(":", 1)
        if parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]), int(parts[1])
        return None
    if token.isdigit():
        return active_folder, int(token)
    return None


def _find_track_matches(
    library: list[dict], token: str, folder_index: int | None
) -> list[tuple[int, int, str]]:
    matches: list[tuple[int, int, str]] = []
    if folder_index is not None:
        folders = [(folder_index, library[folder_index])] if 0 <= folder_index < len(library) else []
    else:
        folders = list(enumerate(library))
    for fidx, entry in folders:
        for tidx, path in enumerate(entry["tracks"]):
            name = Path(path).name
            stem = Path(path).stem
            if token.lower() in (name.lower(), stem.lower()):
                matches.append((fidx, tidx, path))
    return matches


def _resolve_track_selection(
    library: list[dict], active_folder: int, token: str
) -> tuple[str, int | None, int | None] | None:
    if not token:
        return None
    selection = _parse_folder_track(token, active_folder)
    if selection:
        folder_index, track_index = selection
        if folder_index < 0 or folder_index >= len(library):
            print("Folder index out of range.")
            return None
        tracks = library[folder_index]["tracks"]
        if track_index < 0 or track_index >= len(tracks):
            print("Track index out of range.")
            return None
        return tracks[track_index], folder_index, track_index

    path = Path(token)
    if path.exists():
        path_str = str(path)
        for fidx, entry in enumerate(library):
            if path_str in entry["tracks"]:
                return path_str, fidx, entry["tracks"].index(path_str)
        return path_str, None, None

    matches = _find_track_matches(library, token, active_folder)
    if not matches:
        matches = _find_track_matches(library, token, None)
    if len(matches) == 1:
        fidx, tidx, path_str = matches[0]
        return path_str, fidx, tidx
    if len(matches) > 1:
        print("Multiple matches, use 'menu' and choose an index.")
    return None


def _resolve_soundfont_token(controller: PlaybackController, token: str) -> bool:
    if token.isdigit():
        fonts = _list_soundfonts()
        idx = int(token)
        if 0 <= idx < len(fonts):
            return controller.set_soundfont(str(fonts[idx]))
        print("Soundfont index out of range.")
        return False
    path = Path(token)
    if path.exists():
        return controller.set_soundfont(str(path))
    for font in _list_soundfonts():
        if token.lower() == font.name.lower() or token.lower() == font.stem.lower():
            return controller.set_soundfont(str(font))
    return False


def run_cli() -> int:
    library = _build_library()
    active_folder = 0 if library else -1
    controller = PlaybackController(tracks=_folder_tracks(library, active_folder), piano_sink=NullPianoSink())
    print("MIDI Player CLI. Type 'help' for commands.")
    _print_menu(library, active_folder, controller.current_path)
    while True:
        try:
            line = input("cg> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        args = shlex.split(line)
        cmd = args[0].lower()
        rest = args[1:]

        if cmd in ("quit", "exit", "q"):
            break
        if cmd in ("help", "h", "?"):
            _print_help()
            continue
        if cmd in ("menu", "list", "ls"):
            _print_menu(library, active_folder, controller.current_path)
            continue
        if cmd in ("folders", "dirs"):
            _print_folders(library, active_folder)
            continue
        if cmd in ("tracks", "files"):
            if rest and rest[0].isdigit():
                _print_tracks_for_folder(library, int(rest[0]), controller.current_path)
            else:
                _print_tracks_for_folder(library, active_folder, controller.current_path)
            continue
        if cmd == "use":
            if not rest or not rest[0].isdigit():
                print("Usage: use <folder_index>")
                continue
            idx = int(rest[0])
            if idx < 0 or idx >= len(library):
                print("Folder index out of range.")
                continue
            if controller.status().get("status") in ("playing", "paused"):
                controller.stop()
            active_folder = idx
            controller.set_tracks(_folder_tracks(library, active_folder))
            print(f"Active folder: [{active_folder}] {library[active_folder]['label']}")
            continue
        if cmd == "rescan":
            previous_path = None
            if 0 <= active_folder < len(library):
                previous_path = library[active_folder]["path"]
            if controller.status().get("status") in ("playing", "paused"):
                controller.stop()
            library = _build_library()
            active_folder = 0 if library else -1
            if previous_path:
                for idx, entry in enumerate(library):
                    if entry["path"] == previous_path:
                        active_folder = idx
                        break
            controller.set_tracks(_folder_tracks(library, active_folder))
            _print_menu(library, active_folder, controller.current_path)
            continue
        if cmd == "status":
            _print_status(controller)
            continue
        if cmd in ("analyze", "analyse", "info"):
            target = rest[0] if rest else None
            if target and len(rest) >= 2 and rest[0].isdigit() and rest[1].isdigit():
                target = f"{rest[0]}:{rest[1]}"
            if target:
                selection = _resolve_track_selection(library, active_folder, target)
                if not selection:
                    print("Unable to analyze track.")
                    continue
                target = selection[0]
            summary = controller.analyze_track(target)
            if summary:
                _print_analysis(summary)
            else:
                print("Unable to analyze track.")
            continue
        if cmd == "play":
            target = None
            if rest:
                if len(rest) >= 2 and rest[0].isdigit() and rest[1].isdigit():
                    target = f"{rest[0]}:{rest[1]}"
                else:
                    target = rest[0]
            if target:
                selection = _resolve_track_selection(library, active_folder, target)
                if not selection:
                    continue
                path, folder_index, track_index = selection
                if folder_index is not None and folder_index != active_folder:
                    if controller.status().get("status") in ("playing", "paused"):
                        controller.stop()
                    active_folder = folder_index
                    controller.set_tracks(_folder_tracks(library, active_folder))
                if track_index is not None:
                    if not controller.load_track(track_index):
                        continue
                else:
                    if not controller.load_track(path):
                        continue
            else:
                if controller.current_path is None and _folder_tracks(library, active_folder):
                    controller.load_track(0)
            if controller.play():
                current = controller.current_path
                if current:
                    print(f"Playing: {Path(current).name}")
            continue
        if cmd == "pause":
            controller.pause()
            print("Paused.")
            continue
        if cmd == "resume":
            controller.resume()
            print("Resumed.")
            continue
        if cmd == "stop":
            controller.stop()
            print("Stopped.")
            continue
        if cmd == "next":
            if controller.next_track(auto_play=True):
                current = controller.current_path
                if current:
                    print(f"Playing: {Path(current).name}")
            continue
        if cmd == "prev":
            if controller.prev_track(auto_play=True):
                current = controller.current_path
                if current:
                    print(f"Playing: {Path(current).name}")
            continue
        if cmd == "seek":
            if not rest:
                print("Usage: seek <ms>")
                continue
            try:
                target_ms = float(rest[0])
                controller.seek(target_ms)
                print(f"Seeked to {target_ms:.0f} ms.")
            except ValueError:
                print("Invalid seek time.")
            continue
        if cmd in ("soundfonts", "sfonts", "sflist"):
            _print_soundfonts(controller)
            continue
        if cmd in ("soundfont", "sf"):
            if not rest:
                print(f"soundfont: {controller.sound_font}")
                continue
            if _resolve_soundfont_token(controller, rest[0]):
                print(f"soundfont set to: {controller.sound_font}")
            continue

        selection = _resolve_track_selection(library, active_folder, cmd)
        if selection:
            path, folder_index, track_index = selection
            if folder_index is not None and folder_index != active_folder:
                if controller.status().get("status") in ("playing", "paused"):
                    controller.stop()
                active_folder = folder_index
                controller.set_tracks(_folder_tracks(library, active_folder))
            if track_index is not None:
                if not controller.load_track(track_index):
                    continue
            else:
                if not controller.load_track(path):
                    continue
            if controller.play():
                current = controller.current_path
                if current:
                    print(f"Playing: {Path(current).name}")
            continue

        print(f"Unknown command: {cmd}")

    controller.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
