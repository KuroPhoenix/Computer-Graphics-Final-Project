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

from audio.audio_handler import AudioHandler


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


def _print_folders(library: list[dict], active_folder: int) -> None:
    if not library:
        print("No MIDI folders found.")
        return
    print("Folders (use 'use <index>' to select):")
    for entry in library:
        idx = entry["folder_index"]
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
    for track in entry["tracks"]:
        idx = track["track_index"]
        path = track["path"]
        marker = "*" if path == current_path else " "
        name = track["name"]
        print(f"{marker} [{idx}] {name} -> {path}")


def _print_menu(library: list[dict], active_folder: int, current_path: str | None) -> None:
    _print_folders(library, active_folder)
    if library:
        _print_tracks_for_folder(library, active_folder, current_path)


def _print_soundfonts(handler: AudioHandler) -> None:
    fonts = handler.List_Soundfonts()
    if not fonts:
        print("No soundfonts found.")
        return
    print("Soundfonts (use 'soundfont <index>' to switch):")
    for idx, path in enumerate(fonts):
        marker = "*" if str(path) == handler.sound_font else " "
        print(f"{marker} [{idx}] {path.name} -> {path}")


def _print_status(handler: AudioHandler) -> None:
    status = handler.Status()
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


def run_cli() -> int:
    handler = AudioHandler()
    print("MIDI Player CLI. Type 'help' for commands.")
    _print_menu(handler.List_Songs(), handler.active_folder_index, handler.current_path)
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
            _print_menu(handler.List_Songs(), handler.active_folder_index, handler.current_path)
            continue
        if cmd in ("folders", "dirs"):
            _print_folders(handler.List_Songs(), handler.active_folder_index)
            continue
        if cmd in ("tracks", "files"):
            library = handler.List_Songs()
            if rest and rest[0].isdigit():
                _print_tracks_for_folder(library, int(rest[0]), handler.current_path)
            else:
                _print_tracks_for_folder(library, handler.active_folder_index, handler.current_path)
            continue
        if cmd == "use":
            if not rest or not rest[0].isdigit():
                print("Usage: use <folder_index>")
                continue
            idx = int(rest[0])
            if not handler.Set_Folder(idx):
                print("Folder index out of range.")
                continue
            library = handler.List_Songs()
            if 0 <= idx < len(library):
                print(f"Active folder: [{idx}] {library[idx]['label']}")
            continue
        if cmd == "rescan":
            library = handler.Rescan()
            _print_menu(library, handler.active_folder_index, handler.current_path)
            continue
        if cmd == "status":
            _print_status(handler)
            continue
        if cmd in ("analyze", "analyse", "info"):
            target = None
            if rest:
                if len(rest) >= 2 and rest[0].isdigit() and rest[1].isdigit():
                    target = (int(rest[0]), int(rest[1]))
                elif rest[0].isdigit():
                    target = int(rest[0])
                else:
                    target = rest[0]
            summary = handler.Analyze_Song(target)
            if summary:
                _print_analysis(summary)
            else:
                print("Unable to analyze track.")
            continue
        if cmd == "play":
            target = None
            if rest:
                if len(rest) >= 2 and rest[0].isdigit() and rest[1].isdigit():
                    target = (int(rest[0]), int(rest[1]))
                elif rest[0].isdigit():
                    target = int(rest[0])
                else:
                    target = rest[0]
            if handler.Play_Song(target):
                current = handler.current_path
                if current:
                    print(f"Playing: {Path(current).name}")
            continue
        if cmd == "pause":
            handler.Pause_Song()
            print("Paused.")
            continue
        if cmd == "resume":
            handler.Resume_Song()
            print("Resumed.")
            continue
        if cmd == "stop":
            handler.Stop_Song()
            print("Stopped.")
            continue
        if cmd == "next":
            if handler.Next_song():
                current = handler.current_path
                if current:
                    print(f"Playing: {Path(current).name}")
            continue
        if cmd == "prev":
            if handler.Prev_song():
                current = handler.current_path
                if current:
                    print(f"Playing: {Path(current).name}")
            continue
        if cmd == "seek":
            if not rest:
                print("Usage: seek <ms>")
                continue
            try:
                target_ms = float(rest[0])
                handler.Seek(target_ms)
                print(f"Seeked to {target_ms:.0f} ms.")
            except ValueError:
                print("Invalid seek time.")
            continue
        if cmd in ("soundfonts", "sfonts", "sflist"):
            _print_soundfonts(handler)
            continue
        if cmd in ("soundfont", "sf"):
            if not rest:
                print(f"soundfont: {handler.sound_font}")
                continue
            token = rest[0]
            selection: object = token
            if token.isdigit():
                selection = int(token)
            if handler.Set_Soundfont(selection):
                print(f"soundfont set to: {handler.sound_font}")
            continue

        selection: object = cmd
        if cmd.isdigit():
            selection = int(cmd)
        if handler.Play_Song(selection):
            current = handler.current_path
            if current:
                print(f"Playing: {Path(current).name}")
            continue

        print(f"Unknown command: {cmd}")

    handler.Shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
