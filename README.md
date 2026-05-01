# Live Translate 🎙️🌍

**Real-time Automatic Speech Translation for Live Events**

Stream speech through your microphone → auto-translate to 8 languages → audience
hears it in their chosen language on their phone or laptop, with subtitles,
in near-real-time. Fully **offline** — no cloud, no internet required after setup.

---

## Architecture

```
Microphone
    │
    ▼
whisper-stream (whisper.cpp)    ← local STT, models: tiny/base/small/medium/large-v3
    │  raw text segments
    ▼
asyncio Segment Queue
    │
    ▼
Translation Worker              ← argostranslate (offline) OR Gemma 4B/12B (Ollama)
    │  translated text per language
    ▼
Per-language Audio Queue × 8
    │
    ▼
TTS Worker × 8                  ← Piper | Parkiet | Supertonic
    │  WAV audio chunks
    ▼
WebSocket Broadcast
    │
    ├── 🇬🇧 EN listeners (binary WAV + JSON subtitles)
    ├── 🇳🇱 NL listeners
    ├── 🇪🇸 ES listeners
    ├── 🇵🇹 PT listeners
    ├── 🇫🇷 FR listeners
    ├── 🇩🇪 DE listeners
    ├── 🇮🇹 IT listeners
    └── 🇺🇦 UK listeners
```

---

## Quick Start

### 1. Install dependencies

```bash
# macOS prerequisites
brew install sdl2 portaudio cmake

# Ubuntu/Debian prerequisites
sudo apt install libsdl2-dev portaudio19-dev cmake build-essential

# Run automated setup (builds whisper.cpp, downloads models & voices)
bash setup.sh
```

### 2. Start the server

```bash
source .venv/bin/activate
python run.py
# Custom port:
python run.py --port 9000
```

### 3. Open the interfaces

| URL | Who |
|-----|-----|
| `http://YOUR_IP:8765/operator` | Operator / admin |
| `http://YOUR_IP:8765/client`   | Audience (mobile/desktop) |

> **Tip:** Share the client URL as a QR code (shown in the operator panel).
> Audience connects on their phone via the event's local WiFi.

---

## Supported Languages

| Code | Language    | Piper Voice (default)         |
|------|-------------|-------------------------------|
| `en` | English     | en_US-lessac-medium           |
| `nl` | Dutch       | nl_NL-mls-medium              |
| `es` | Spanish     | es_ES-mls-medium              |
| `pt` | Portuguese  | pt_BR-edresson-low            |
| `fr` | French      | fr_FR-mls-medium              |
| `de` | German      | de_DE-thorsten-medium         |
| `it` | Italian     | it_IT-riccardo-x_low          |
| `uk` | Ukrainian   | uk_UA-ukrainian_tts-medium    |

---

## STT — whisper.cpp (whisper-stream)

whisper-stream listens continuously to the selected microphone and outputs
transcribed segments as plain text to stdout. The server reads these in real-time.

**Models** (place `.bin` files in `models/`):

```bash
# Download any model from HuggingFace
curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin" \
     -o models/ggml-small.bin
```

| Model     | Size   | Speed  | Accuracy |
|-----------|--------|--------|----------|
| tiny      | 75 MB  | ⚡⚡⚡  | ★★       |
| base      | 142 MB | ⚡⚡⚡  | ★★★      |
| small     | 466 MB | ⚡⚡    | ★★★★     |
| medium    | 1.5 GB | ⚡     | ★★★★★    |
| large-v3  | 3.1 GB | 🐢    | ★★★★★★   |

---

## Translation Engines

### Argostranslate (recommended — fully offline)
```bash
# Packages are auto-downloaded on first run (or via setup.sh)
pip install argostranslate
```

### Gemma 4B / 12B via Ollama
```bash
# Install Ollama: https://ollama.com
ollama pull gemma3:4b    # ~2.3 GB, good quality
ollama pull gemma3:12b   # ~7 GB,  best quality
# Start Ollama server: ollama serve
```
Configure the Ollama host in the operator panel (default: `http://localhost:11434`).

---

## TTS Engines

### Piper (recommended — all languages)
- Download: https://github.com/rhasspy/piper/releases
- Place binary on `PATH` or set `PIPER_BIN=/path/to/piper`
- Voice models (`.onnx`): place in `voices/<lang_code>/`

```
voices/
├── en/   en_US-lessac-medium.onnx
│         en_US-lessac-medium.onnx.json
├── nl/   nl_NL-mls-medium.onnx
├── es/   es_ES-mls-medium.onnx
...
```

Piper voices: https://huggingface.co/rhasspy/piper-voices

### Parkiet
- Install: https://github.com/pevers/parkiet
- Set `PARKIET_BIN=/path/to/parkiet`
- Place models in `voices/parkiet_<lang>/`

### Supertonic (es/pt/fr)
- Install: `pip install git+https://github.com/supertone-inc/supertonic.git`
- Falls back to Piper for languages not in {es, pt, fr}
- Place models in `voices/supertonic_<lang>/`

---

## Voice Overrides

Place multiple voice files per language in the `voices/<lang>/` folder.
The operator panel shows all detected voices in the **Voice Overrides** tab
and lets you select one per language. The selection is applied live.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPER_BIN` | `piper` | Path to Piper binary |
| `PARKIET_BIN` | `parkiet` | Path to Parkiet binary |
| `WHISPER_STREAM_BIN` | `whisper.cpp/build/bin/whisper-stream` | Path to whisper-stream |

---

## Operator Panel Features

- **Start / Stop** broadcast with one click
- Choose **Whisper model**, **source language**, **microphone**
- Switch **translation engine** (Argos / Gemma 4B / Gemma 12B) live
- Switch **TTS engine** (Piper / Parkiet / Supertonic) live
- Adjust **voice speed** (0.5× – 2.0×)
- Per-language **voice override** dropdown (auto-detects all files in `voices/<lang>/`)
- Live **transcript feed** of everything recognised
- **Listener count** per language in real-time
- **System log**

---

## Client Portal (Audience)

- Mobile-first, PWA-ready
- Language selection screen with flags
- Live **translated subtitles** displayed in large readable text
- **Original speech** shown simultaneously
- **Audio playback** via Web Audio API (user-initiated, works on iOS/Android)
- Automatic WebSocket reconnection
- Language can be switched at any time

---

## Performance — 500–1000 Users

- FastAPI + `uvicorn` handles 1000 concurrent WebSocket connections on a modern CPU
- Audio is synthesised **once per language**, then broadcast to all N subscribers simultaneously
- `asyncio.gather` fan-out — no per-user TTS cost
- Bounded queues prevent memory blowup under load
- For > 500 users on a single machine: use `nginx` as a WebSocket proxy/load-balancer
  in front of multiple `uvicorn` workers (note: WebSocket state requires sticky sessions
  or a Redis pub-sub bridge between workers)

---

## Project Structure

```
live-translate/
├── backend/
│   ├── main.py              FastAPI app, all routes + WebSocket endpoints
│   ├── config.py            Settings, language definitions, paths
│   ├── pipeline.py          STT → Translation → TTS orchestrator (asyncio queues)
│   ├── broadcast.py         WebSocket connection manager
│   ├── stt/
│   │   └── whisper_manager.py  whisper-stream subprocess wrapper
│   ├── translation/
│   │   ├── argos_engine.py     Argostranslate (offline)
│   │   └── gemma_engine.py     Gemma via Ollama
│   └── tts/
│       ├── piper_engine.py     Piper TTS
│       ├── parkiet_engine.py   Parkiet TTS
│       └── supertonic_engine.py  Supertonic TTS
├── frontend/
│   ├── operator.html        Operator control panel
│   └── client.html          Audience listener portal
├── voices/                  TTS voice model files (organised by language)
├── models/                  Whisper .bin model files
├── whisper.cpp/             whisper.cpp source (cloned by setup.sh)
├── requirements.txt
├── setup.sh
└── run.py
```

---

## License

MIT — use freely for live events, conferences, churches, and community gatherings.
