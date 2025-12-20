MIDI_FILES = ["midi/tracks/natlan.mid", "midi/tracks/Inazuma.mid", "midi/tracks/Moon in one's cup.mid"]
# MIDI discovery
MIDI_SCAN_ROOT = "midi"
MIDI_SCAN_RECURSIVE = True
MIDI_EXTENSIONS = (".mid", ".midi")

# Logging configuration
LOG_LEVEL = "DEBUG"
LOG_FILE = "logs/app.log"
# Disable console logging for CLI; logs remain in LOG_FILE.
LOG_CONSOLE = False
# Suppress stderr output in CLI (redirect to logs/cli_stderr.log).
CLI_SUPPRESS_STDERR = True

# Audio configuration
# Options: "pyfluidsynth" (per-note via FluidSynth), "midi" (pygame.midi), "mixer" (play full file), "none"
AUDIO_BACKEND = "pyfluidsynth"
# Optional explicit MIDI device id; None uses default (only for "midi" backend)
AUDIO_MIDI_DEVICE_ID = None
# Path to a SoundFont file (used by pyfluidsynth); FluidR3 installs here by default
SOUND_FONT_PATH = "midi/soundfonts/salamander.sf2"
# Overall volume for pyfluidsynth (0.0–2.0). Lower reduces clipping/static.
FLUID_GAIN = 0.3

# Optional path to FluidSynth DLLs on Windows (add this directory before importing pyfluidsynth)
FLUIDSYNTH_DLL_PATH = ""
# Explicit FluidSynth driver to request; use "auto" to pick a sensible default per-OS
# Common values: pulseaudio (WSL/Linux), dsound/wasapi/portaudio/sdl2 (Windows)
FLUID_DRIVER = "auto"
# Low-level audio timing knobs for FluidSynth (PyFluidSynthOutput)
FLUID_SAMPLE_RATE = 44100.0
# Increase buffer size slightly to reduce crackle on real-time output.
FLUID_PERIOD_SIZE = 512
FLUID_PERIODS = 4
# Optional override for estimated FluidSynth output latency (ms); None uses derived value.
FLUID_LATENCY_MS = None
# FluidSynth quality/stability options
# Interpolation quality (0=none, 4=7th order). Set to None if unsupported by your FluidSynth build.
FLUID_INTERP = None
# Disable reverb/chorus to reduce noise/CPU unless you explicitly want them.
FLUID_REVERB = False
FLUID_CHORUS = False
# Cap polyphony to avoid CPU spikes on dense passages.
FLUID_POLYPHONY = 256
# Pulseaudio target latency (ms); applied when using the pulseaudio driver.
PULSE_LATENCY_MSEC = 60
# Optional MIDI driver override for FluidSynth (Windows builds may only support "winmidi").
FLUID_MIDI_DRIVER = None
# Suppress FluidSynth logging in the CLI.
FLUID_SUPPRESS_LOGS = True

# Instrument policy for limited (e.g., piano-only) soundfonts.
# When True, ignore program changes and force a single preset on all channels.
FORCE_PROGRAM = True
FORCE_PROGRAM_BANK = 0
FORCE_PROGRAM_PRESET = 0
# Enable GM drum bank (128) on channel 10 when using GM soundfonts.
ENABLE_GM_DRUMS = False

# Scheduler tuning
# Poll interval when paused/idle; smaller improves responsiveness at the cost of CPU.
SCHEDULER_POLL_MS = 5.0
# Busy-wait window for sub-millisecond precision near events.
SCHEDULER_SPIN_MS = 0.5

# Audio latency compensation (ms); added to backend estimates before scheduling.
AUDIO_LATENCY_MS = 0.0
# Expected MIDI device latency (ms) for pygame.midi backend.
MIDI_LATENCY_MS = 0.0

# Per-track auto profiling (analysis-driven tweaks before playback)
AUTO_PROFILE = True
TARGET_AVG_VELOCITY = 90.0
VELOCITY_SCALE_MIN = 0.7
VELOCITY_SCALE_MAX = 1.3
AUTO_GAIN = True
POLYPHONY_TARGET = 8.0
GAIN_MIN = 0.2
GAIN_MAX = 0.5
AUTO_CC7_SCALE = False
TARGET_CC7 = 100.0
CC7_SCALE_MIN = 0.8
CC7_SCALE_MAX = 1.2
AUTO_CC11_SCALE = False
TARGET_CC11 = 100.0
CC11_SCALE_MIN = 0.8
CC11_SCALE_MAX = 1.2
