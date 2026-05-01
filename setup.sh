#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Live Translate — Setup Script
# Run once from the project root:   bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $1"; }

echo ""
echo "═══════════════════════════════════════════════"
echo "  Live Translate — Setup"
echo "═══════════════════════════════════════════════"
echo ""

# ── 1. Python venv ────────────────────────────────
info "Setting up Python virtual environment…"
[ ! -d ".venv" ] && python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
success "Python deps installed"

# ── 2. Directories ────────────────────────────────
info "Creating voice/model directories…"
mkdir -p models voices/{en,nl,es,pt,fr,de,it,uk} whisper.cpp/build/bin
success "Directories ready"

# ── 3. whisper.cpp ────────────────────────────────
info "Checking whisper.cpp…"
if [ ! -f "whisper.cpp/CMakeLists.txt" ]; then
  info "Cloning whisper.cpp…"
  git clone https://github.com/ggml-org/whisper.cpp.git --depth 1
fi
if [ ! -f "whisper.cpp/build/bin/whisper-stream" ]; then
  info "Building whisper-stream (requires SDL2: brew install sdl2 / apt install libsdl2-dev)…"
  cd whisper.cpp
  cmake -B build -DWHISPER_SDL2=ON 2>&1 | tail -3
  cmake --build build --config Release \
    -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 2)" 2>&1 | tail -5
  cd ..
  if [ -f "whisper.cpp/build/bin/whisper-stream" ]; then
    success "whisper-stream built"
  else
    warn "whisper-stream build failed — install SDL2 first then re-run setup.sh"
  fi
else
  success "whisper-stream already built"
fi

# ── 4. Whisper model ──────────────────────────────
info "Downloading whisper 'base' model (142 MB)…"
if [ ! -f "models/ggml-base.bin" ]; then
  curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" \
       -o "models/ggml-base.bin" --progress-bar
  success "Model: models/ggml-base.bin"
else
  success "whisper base model already present"
fi

# ── 5. Piper binary ───────────────────────────────
info "Checking Piper TTS binary…"
if ! command -v piper &>/dev/null && [ ! -f ".venv/bin/piper" ]; then
  PIPER_VER="1.2.0"
  UNAME=$(uname -s); ARCH=$(uname -m)
  if   [[ "$UNAME" == "Darwin" && "$ARCH" == "arm64" ]]; then
    URL="https://github.com/rhasspy/piper/releases/download/v${PIPER_VER}/piper_macos_aarch64.tar.gz"
  elif [[ "$UNAME" == "Darwin" ]]; then
    URL="https://github.com/rhasspy/piper/releases/download/v${PIPER_VER}/piper_macos_x64.tar.gz"
  elif [[ "$ARCH" == "aarch64" ]]; then
    URL="https://github.com/rhasspy/piper/releases/download/v${PIPER_VER}/piper_linux_aarch64.tar.gz"
  else
    URL="https://github.com/rhasspy/piper/releases/download/v${PIPER_VER}/piper_linux_x86_64.tar.gz"
  fi
  info "Downloading Piper from $URL"
  if curl -sL "$URL" | tar xz -C /tmp/ 2>/dev/null; then
    cp /tmp/piper/piper .venv/bin/piper 2>/dev/null && chmod +x .venv/bin/piper \
      && success "Piper installed to .venv/bin/piper" \
      || warn "Could not install Piper — download manually from https://github.com/rhasspy/piper/releases"
  else
    warn "Could not download Piper — download manually from https://github.com/rhasspy/piper/releases"
  fi
else
  success "Piper already available"
fi

# ── 6. Piper voice models ─────────────────────────
# Downloads directly into voices/<lang>/ with no subdirectories.
# The piper_engine.py also does recursive scan as a safety net.
info "Downloading Piper voice models…"

# Base URL on HuggingFace (voices are stored at lang/sub/name.onnx)
HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"

dl_voice() {
  local lang_code="$1"       # e.g. "es"
  local hf_path="$2"         # e.g. "es/es_ES/es_ES-mls-medium"
  local voice_name           # derived from hf_path
  voice_name="$(basename "$hf_path")"
  local dest_dir="voices/${lang_code}"
  local onnx="${dest_dir}/${voice_name}.onnx"
  local json="${dest_dir}/${voice_name}.onnx.json"

  mkdir -p "$dest_dir"

  if [ -f "$onnx" ]; then
    success "Voice exists : ${onnx}"
    return
  fi

  info "Downloading voice ${voice_name} → voices/${lang_code}/"
  if curl -fL "${HF_BASE}/${hf_path}.onnx"      -o "$onnx" --progress-bar 2>/dev/null \
  && curl -fsL "${HF_BASE}/${hf_path}.onnx.json" -o "$json" 2>/dev/null; then
    success "Voice OK : ${onnx}"
  else
    rm -f "$onnx" "$json"
    warn "Could not download voice ${voice_name} — check HuggingFace or add .onnx manually to ${dest_dir}/"
  fi
}

# Download each voice directly into voices/<lang>/ — flat, no nesting
dl_voice "en" "en/en_US/en_US-lessac-medium"
dl_voice "nl" "nl/nl_NL/nl_NL-mls-medium"
dl_voice "es" "es/es_ES/es_ES-mls-medium"
dl_voice "pt" "pt/pt_BR/pt_BR-edresson-low"
dl_voice "fr" "fr/fr_FR/fr_FR-mls-medium"
dl_voice "de" "de/de_DE/de_DE-thorsten-medium"
dl_voice "it" "it/it_IT/it_IT-riccardo-x_low"
dl_voice "uk" "uk/uk_UA/uk_UA-ukrainian_tts-medium"

# Print what we found
echo ""
info "Voice files discovered:"
for lang in en nl es pt fr de it uk; do
  count=$(find "voices/${lang}" -name "*.onnx" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$count" -gt 0 ]; then
    echo -e "  ${GREEN}✓${NC} voices/${lang}/ — ${count} model(s)"
    find "voices/${lang}" -name "*.onnx" | sed "s/^/      /"
  else
    echo -e "  ${YELLOW}✗${NC} voices/${lang}/ — no .onnx found (TTS will be silent for this language)"
  fi
done
echo ""

# ── 7. Argostranslate packages ────────────────────
info "Pre-downloading argostranslate language packages…"
python3 - <<'PYEOF'
import sys
try:
    import argostranslate.package
    import argostranslate.translate
except ImportError:
    print("  [ERROR] argostranslate not installed — run: pip install argostranslate")
    sys.exit(0)

print("  Updating package index…")
try:
    argostranslate.package.update_package_index()
except Exception as e:
    print(f"  [WARN] Could not update index: {e}")

pairs = [
    ("en","nl"),("en","es"),("en","pt"),("en","fr"),
    ("en","de"),("en","it"),("en","uk"),
    ("nl","en"),("es","en"),
]

try:
    available = argostranslate.package.get_available_packages()
except Exception as e:
    print(f"  [WARN] Could not fetch available packages: {e}")
    available = []

for src, tgt in pairs:
    try:
        installed = argostranslate.translate.get_installed_languages()
        already = any(
            getattr(lang, 'code', str(lang)) == src
            and any(
                getattr(t, 'code', None) == tgt
                or getattr(getattr(t, 'language', None), 'code', None) == tgt
                for t in lang.translations_to
            )
            for lang in installed
        )
        if already:
            print(f"  ✓ {src}->{tgt} already installed")
            continue
    except Exception:
        pass  # will attempt install anyway

    pkg = next((p for p in available if p.from_code==src and p.to_code==tgt), None)
    if pkg:
        print(f"  Installing {src}->{tgt}…")
        try:
            argostranslate.package.install_from_path(pkg.download())
            print(f"  ✓ {src}->{tgt} installed")
        except Exception as e:
            print(f"  [WARN] Failed to install {src}->{tgt}: {e}")
    else:
        print(f"  [WARN] No package found for {src}->{tgt} — will use English pivot at runtime")
print("  Done.")
PYEOF
success "Argostranslate packages ready"

# ── Done ──────────────────────────────────────────
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""
echo "  Start the server:"
echo "    source .venv/bin/activate && python run.py"
echo ""
echo "  Diagnose whisper output:  python run.py --debug"
echo ""
echo "  Operator : http://localhost:8765/operator"
echo "  Audience : http://localhost:8765/client"
echo ""
echo "  Optional Gemma translation:"
echo "    brew install ollama && ollama pull gemma3:4b && ollama serve"
echo ""
