"""
Translation Engine — Argostranslate
Fully offline neural translation.
Packages are downloaded once from the internet then cached locally.
https://github.com/argosopentech/argos-translate
"""
import asyncio
import logging

from .base import BaseTranslator

logger = logging.getLogger(__name__)


class ArgosTranslator(BaseTranslator):
    name = "argos"

    def __init__(self):
        self._installed: set = set()

    # ------------------------------------------------------------------
    async def setup(self) -> None:
        """Pre-download packages for every language pair we might need."""
        logger.info("ArgosTranslate: updating package index…")
        try:
            await asyncio.to_thread(self._update_index)
            pairs = self._pairs_needed()
            for pair in pairs:
                await self._ensure_package(*pair)
            logger.info("ArgosTranslate: ready (%d pairs)", len(pairs))
        except Exception as exc:
            logger.warning("ArgosTranslate setup failed: %s", exc)

    def _update_index(self):
        import argostranslate.package
        argostranslate.package.update_package_index()

    def _pairs_needed(self):
        target_langs = ["nl", "es", "pt", "fr", "de", "it", "uk", "en"]
        source_langs  = ["en", "nl", "es"]
        pairs = set()
        for src in source_langs:
            for tgt in target_langs:
                if src != tgt:
                    pairs.add((src, tgt))
                # Argos routes via English; make sure en↔X exists
                if src != "en":
                    pairs.add(("en", tgt))
                    pairs.add((src, "en"))
        return pairs

    async def _ensure_package(self, src: str, tgt: str) -> None:
        if (src, tgt) in self._installed:
            return

        def _get_lang_code(lang_obj) -> str:
            """
            Safely extract language code from an installed-language object.
            Older argostranslate uses .code; newer builds wrap it as .language or
            expose it via str() or .__str__().
            """
            for attr in ("code", "language", "lang"):
                val = getattr(lang_obj, attr, None)
                if val is not None:
                    return str(val)
            return str(lang_obj)

        def _get_translation_code(t_obj) -> str:
            """Same safe accessor for a CachedTranslation / ITranslation target."""
            for attr in ("code", "language", "to_language", "to_code"):
                val = getattr(t_obj, attr, None)
                if val is not None:
                    # May itself be an object with a .code attr
                    inner = getattr(val, "code", None) or getattr(val, "language", None)
                    return str(inner) if inner else str(val)
            return str(t_obj)

        def _install():
            import argostranslate.package
            import argostranslate.translate
            try:
                installed = argostranslate.translate.get_installed_languages()
                already = any(
                    _get_lang_code(lang) == src
                    and any(_get_translation_code(t) == tgt for t in lang.translations_to)
                    for lang in installed
                )
            except Exception as exc:
                logger.debug("Could not check installed languages: %s — will attempt install", exc)
                already = False

            if already:
                return
            available = argostranslate.package.get_available_packages()
            pkg = next(
                (p for p in available if p.from_code == src and p.to_code == tgt),
                None,
            )
            if pkg:
                argostranslate.package.install_from_path(pkg.download())
                logger.info("ArgosTranslate: installed %s→%s", src, tgt)
            else:
                logger.warning("ArgosTranslate: no package for %s→%s", src, tgt)

        await asyncio.to_thread(_install)
        self._installed.add((src, tgt))

    # ------------------------------------------------------------------
    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            return text
        await self._ensure_package(source_lang, target_lang)

        def _do():
            import argostranslate.translate
            return argostranslate.translate.translate(text, source_lang, target_lang)

        try:
            result = await asyncio.to_thread(_do)
            return result or text
        except Exception as exc:
            logger.error("ArgosTranslate error (%s→%s): %s", source_lang, target_lang, exc)
            # Try going via English as pivot if direct pair fails
            if source_lang != "en" and target_lang != "en":
                try:
                    logger.info("Retrying via English pivot: %s→en→%s", source_lang, target_lang)
                    await self._ensure_package(source_lang, "en")
                    await self._ensure_package("en", target_lang)

                    def _pivot():
                        import argostranslate.translate
                        en_text = argostranslate.translate.translate(text, source_lang, "en")
                        return argostranslate.translate.translate(en_text, "en", target_lang)

                    result = await asyncio.to_thread(_pivot)
                    return result or text
                except Exception as exc2:
                    logger.error("Pivot translate also failed: %s", exc2)
            return text
