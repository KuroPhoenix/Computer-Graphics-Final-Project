MIDI_FILES = ["midi/Inazuma.mid", "midi/Moon in one's cup.mid"]

# Logging configuration
LOG_LEVEL = "DEBUG"
LOG_FILE = "logs/app.log"

# Audio configuration
# Options: "pyfluidsynth" (per-note via FluidSynth), "midi" (pygame.midi), "mixer" (play full file), "none"
AUDIO_BACKEND = "pyfluidsynth"
# Optional explicit MIDI device id; None uses default (only for "midi" backend)
AUDIO_MIDI_DEVICE_ID = None
# Path to a SoundFont file (used by pyfluidsynth); FluidR3 installs here by default
SOUND_FONT_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
# Overall volume for pyfluidsynth (0.0–2.0)
FLUID_GAIN = 0.8
