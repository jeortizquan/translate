"""
Profanity / Content Filter
===========================
Cleans STT segments before they reach translation or TTS.

Two sources of blocked words:
  1. bad_words.txt  — default list bundled with the project (one word/phrase per line)
  2. custom_words.txt — operator-managed additions, editable via the API and operator UI

Both files live in the project root (next to run.py).
Changes to custom_words.txt take effect immediately on the next segment —
no restart required.

Replacement strategy: matched words are replaced with "****" by default,
or with a configurable replacement string.
Matching is case-insensitive and whole-word aware.
"""
import logging
import re
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths (relative to project root — resolved at import time)
_HERE           = Path(__file__).parent.parent   # project root
BAD_WORDS_FILE  = _HERE / "bad_words.txt"
CUSTOM_WORDS_FILE = _HERE / "custom_words.txt"

REPLACEMENT = "****"


class ProfanityFilter:
    """
    Thread-safe profanity filter.
    Reloads word lists on every call to clean() so operator edits
    are picked up immediately without restarting the server.
    """

    def __init__(self):
        self._lock    = threading.Lock()
        self._pattern: re.Pattern | None = None
        self._words:   list[str] = []
        self._mtime_bad    = 0.0
        self._mtime_custom = 0.0

        # Create files if missing
        if not BAD_WORDS_FILE.exists():
            BAD_WORDS_FILE.write_text(_DEFAULT_BAD_WORDS, encoding="utf-8")
            logger.info("Filter: created %s", BAD_WORDS_FILE)

        if not CUSTOM_WORDS_FILE.exists():
            CUSTOM_WORDS_FILE.write_text(
                "# Add custom words/phrases to block, one per line.\n"
                "# Lines starting with # are comments.\n",
                encoding="utf-8",
            )
            logger.info("Filter: created %s", CUSTOM_WORDS_FILE)

        self._reload()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _files_changed(self) -> bool:
        mt_bad    = BAD_WORDS_FILE.stat().st_mtime    if BAD_WORDS_FILE.exists()    else 0.0
        mt_custom = CUSTOM_WORDS_FILE.stat().st_mtime if CUSTOM_WORDS_FILE.exists() else 0.0
        return mt_bad != self._mtime_bad or mt_custom != self._mtime_custom

    def _reload(self) -> None:
        words: set[str] = set()

        def _load(path: Path) -> None:
            if not path.exists():
                return
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    words.add(line.lower())

        _load(BAD_WORDS_FILE)
        _load(CUSTOM_WORDS_FILE)

        self._words = sorted(words, key=len, reverse=True)  # longest first
        self._mtime_bad    = BAD_WORDS_FILE.stat().st_mtime    if BAD_WORDS_FILE.exists()    else 0.0
        self._mtime_custom = CUSTOM_WORDS_FILE.stat().st_mtime if CUSTOM_WORDS_FILE.exists() else 0.0

        if self._words:
            # Build a single compiled regex: \b(word1|word2|...)\b
            # Escape each word; sort by length desc so longer phrases match first
            escaped = [re.escape(w) for w in self._words]
            pattern = r"\b(" + "|".join(escaped) + r")\b"
            self._pattern = re.compile(pattern, re.IGNORECASE)
            logger.debug("Filter: loaded %d word(s)", len(self._words))
        else:
            self._pattern = None
            logger.debug("Filter: word list is empty")

    # ── Public API ────────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """
        Return text with all blocked words replaced by REPLACEMENT.
        Reloads word lists automatically if either file has changed on disk.
        """
        with self._lock:
            if self._files_changed():
                self._reload()
            if self._pattern is None:
                return text
            cleaned = self._pattern.sub(REPLACEMENT, text)
            if cleaned != text:
                logger.debug("Filter: censored %r → %r", text, cleaned)
            return cleaned

    def list_words(self) -> dict:
        """Return current word lists for the operator UI."""
        with self._lock:
            if self._files_changed():
                self._reload()

        bad: list[str] = []
        custom: list[str] = []

        def _read_words(path: Path) -> list[str]:
            if not path.exists():
                return []
            return [
                l.strip() for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.strip().startswith("#")
            ]

        return {
            "bad_words":    _read_words(BAD_WORDS_FILE),
            "custom_words": _read_words(CUSTOM_WORDS_FILE),
            "total":        len(self._words),
        }

    def add_custom_word(self, word: str) -> bool:
        """Append a word to custom_words.txt. Returns True if added, False if already present."""
        word = word.strip().lower()
        if not word or word.startswith("#"):
            return False
        existing = [
            l.strip().lower()
            for l in CUSTOM_WORDS_FILE.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        if word in existing:
            return False
        with open(CUSTOM_WORDS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{word}\n")
        with self._lock:
            self._reload()
        logger.info("Filter: added custom word %r", word)
        return True

    def remove_custom_word(self, word: str) -> bool:
        """Remove a word from custom_words.txt. Returns True if removed."""
        word = word.strip().lower()
        if not CUSTOM_WORDS_FILE.exists():
            return False
        lines = CUSTOM_WORDS_FILE.read_text(encoding="utf-8").splitlines()
        new_lines = [
            l for l in lines
            if l.strip().lower() != word
        ]
        if len(new_lines) == len(lines):
            return False
        CUSTOM_WORDS_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        with self._lock:
            self._reload()
        logger.info("Filter: removed custom word %r", word)
        return True

    def save_custom_words(self, words: list[str]) -> None:
        """Overwrite custom_words.txt with the given list."""
        header = (
            "# Custom blocked words — one per line.\n"
            "# Lines starting with # are comments.\n"
        )
        body = "\n".join(w.strip().lower() for w in words if w.strip() and not w.strip().startswith("#"))
        CUSTOM_WORDS_FILE.write_text(header + body + "\n", encoding="utf-8")
        with self._lock:
            self._reload()
        logger.info("Filter: saved %d custom word(s)", len(words))

    @property
    def enabled(self) -> bool:
        return self._pattern is not None


# ── Singleton ─────────────────────────────────────────────────────────────────
profanity_filter = ProfanityFilter()


# ── Default word list ─────────────────────────────────────────────────────────
_DEFAULT_BAD_WORDS = """\
# Default profanity list — edit freely or add words to custom_words.txt instead.
# Lines starting with # are comments. One word or phrase per line.
ass
asshole
bastard
bitch
bullshit
cock
crap
cunt
damn
dick
dickhead
dumbass
dumbfuck
fuck
fucker
fucking
goddamn
hell
horseshit
idiot
jackass
jerk
motherfucker
nigga
nigger
piss
prick
pussy
shit
shithead
slut
son of a bitch
twat
wanker
whore
"""
