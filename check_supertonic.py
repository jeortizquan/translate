# Init — one instance, shared across all calls
from supertonic import TTS
tts = TTS(model='supertonic-2')          # auto-downloads model on first use

# Load voice style from your JSON files
style = tts.get_voice_style_from_path("voices/supertonic/preset_voice_m.json")
# or the female voice:
style = tts.get_voice_style_from_path("voices/supertonic/preset_voice_f.json")

# Synthesize — returns (wav_array, _)
wav, _ = tts.synthesize(
    text        = "Hello world",
    voice_style = style,
    speed       = 1.05,      # 1.0 = normal, >1 = faster (note: supertonic's default is 1.05)
    lang        = "en",      # any of the 8 supported languages
    total_steps = 5,
    silence_duration = 0.3,
)
# wav is float32 numpy array → convert to int16 → wrap in WAV header