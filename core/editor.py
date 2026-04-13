"""
core/editor.py — постобработка текста через Phi-3.5 Mini (llama-cpp-python).

Что делает LLM (из документации, раздел 8):
  - Убирает слова-паразиты: «ну», «вот», «как бы», «значит», «э-э», «ммм»
  - Расставляет запятые, точки, знаки вопроса и восклицания
  - Делает заглавными первые буквы предложений и имён собственных
  - Разбивает на абзацы по смыслу если текст длиннее 3 предложений
  - Исправляет явные ошибки распознавания (контекстно)
  - НЕ меняет смысл, НЕ добавляет слова от себя
  - Язык вывода совпадает с языком ввода

Параметры (из config.yaml → editor):
  enabled      — false = пропустить LLM, вернуть сырой текст
  model_path   — путь к GGUF файлу Phi-3.5 Mini
  gpu_layers   — -1 = все слои на GPU
  context_size — 2048
  max_tokens   — 1024
  temperature  — 0.1 (низкая = детерминированный вывод)
  timeout_sec  — 10 (таймаут; при превышении — вернуть оригинал)

Консольный тест:
  python -m core.editor
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Системный промпт ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a professional text editor for voice transcription output.

Your task: clean up raw speech-to-text text according to these rules:

1. REMOVE filler words and speech disfluencies:
   - Russian: ну, вот, как бы, значит, э-э, эм, ммм, короче, это самое, блин, ладно
   - English: uh, um, like, you know, basically, right, so, well, I mean, kind of
2. ADD correct punctuation: periods, commas, question marks, exclamation marks.
3. CAPITALIZE first letters of sentences and proper nouns.
4. If the text has more than 3 sentences — split into logical paragraphs.
5. FIX obvious speech recognition errors using context (e.g. "по чему" → "почему").
6. PRESERVE the original language exactly (Russian in → Russian out, English in → English out).
7. Do NOT add new information. Do NOT change meaning. Do NOT translate.

Return ONLY the cleaned text. No explanations, no comments, no prefixes like "Here is..."."""


# ── Результат постобработки ───────────────────────────────────────────────────

@dataclass
class EditResult:
    """Результат одного вызова Editor.edit()."""

    text: str             # Итоговый текст (отредактированный или оригинал)
    original: str         # Исходный текст до обработки
    success: bool         # True = LLM обработал; False = использован оригинал
    reason: str           # Причина fallback ("ok" | "disabled" | "timeout" | "error" | "empty" | "garbled")
    edit_time_sec: float  # Время работы LLM (сек)

    @property
    def was_edited(self) -> bool:
        return self.success and self.text != self.original

    def __str__(self) -> str:
        tag = "LLM" if self.success else f"ORIG({self.reason})"
        return f"[{tag} {self.edit_time_sec:.2f}s] {self.text}"


# ── Редактор ──────────────────────────────────────────────────────────────────

class Editor:
    """
    Обёртка над llama_cpp.Llama (Phi-3.5 Mini Q4_K_M).

    Модель загружается лениво: через load() при старте или автоматически
    при первом вызове edit().

    При любой ошибке или таймауте метод edit() возвращает оригинальный текст
    с success=False — pipeline никогда не теряет результат транскрипции.
    """

    # Если LLM вернул текст длиннее оригинала в MAX_EXPANSION раз — это «мусор»
    MAX_EXPANSION: float = 3.0
    # Минимальная длина входного текста для обращения к LLM (символов)
    MIN_INPUT_CHARS: int = 5

    def __init__(self, config: dict) -> None:
        cfg = config["editor"]

        self.enabled:      bool  = cfg.get("enabled", True)
        self.model_path:   str   = cfg.get("model_path", "models/Phi-3.5-mini-Q4.gguf")
        self.gpu_layers:   int   = cfg.get("gpu_layers", -1)
        self.context_size: int   = cfg.get("context_size", 2048)
        self.max_tokens:   int   = cfg.get("max_tokens", 1024)
        self.temperature:  float = cfg.get("temperature", 0.1)
        self.timeout_sec:  float = cfg.get("timeout_sec", 10.0)

        self._llm = None   # llama_cpp.Llama

    # ── Загрузка ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Загружает Phi-3.5 Mini в VRAM.
        Безопасно вызывать повторно.
        Если editor.enabled=false — ничего не делает.
        """
        if not self.enabled or self._llm is not None:
            return

        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python не установлен. Запустите setup.py."
            ) from exc

        model_file = self._resolve_model_path()

        self._llm = Llama(
            model_path=str(model_file),
            n_gpu_layers=self.gpu_layers,
            n_ctx=self.context_size,
            verbose=False,
            # Phi-3.5 поддерживает chat_format=phi3; llama-cpp определяет авто
        )

    def _resolve_model_path(self) -> Path:
        """Возвращает абсолютный путь к GGUF, проверяет существование."""
        p = Path(self.model_path)
        if not p.is_absolute():
            p = Path(__file__).parent.parent / p
        if not p.exists():
            raise FileNotFoundError(
                f"Модель Phi-3.5 Mini не найдена: {p}\n"
                "Запустите setup.py для скачивания."
            )
        return p

    def _ensure_loaded(self) -> None:
        if self._llm is None and self.enabled:
            self.load()

    # ── Постобработка ─────────────────────────────────────────────────────────

    def edit(self, text: str, language: Optional[str] = None) -> EditResult:
        """
        Постобрабатывает текст через Phi-3.5 Mini.

        Аргументы:
            text     — сырая транскрипция от Whisper
            language — язык ('ru', 'en', …); используется только в промпте

        Возвращает EditResult.
        При disabled / timeout / ошибке — возвращает оригинальный текст.
        """
        # Быстрый путь: LLM отключён
        if not self.enabled:
            return EditResult(
                text=text, original=text,
                success=False, reason="disabled", edit_time_sec=0.0,
            )

        # Слишком короткий ввод — не стоит гонять LLM
        stripped = text.strip()
        if len(stripped) < self.MIN_INPUT_CHARS:
            return EditResult(
                text=stripped, original=text,
                success=False, reason="too_short", edit_time_sec=0.0,
            )

        self._ensure_loaded()

        t_start = time.perf_counter()

        try:
            result_text = self._call_with_timeout(stripped, language)
        except concurrent.futures.TimeoutError:
            return EditResult(
                text=stripped, original=text,
                success=False, reason="timeout",
                edit_time_sec=time.perf_counter() - t_start,
            )
        except Exception as exc:  # noqa: BLE001
            return EditResult(
                text=stripped, original=text,
                success=False, reason=f"error: {exc}",
                edit_time_sec=time.perf_counter() - t_start,
            )

        elapsed = time.perf_counter() - t_start
        result_text = result_text.strip()

        # Валидация результата
        if not result_text:
            return EditResult(
                text=stripped, original=text,
                success=False, reason="empty", edit_time_sec=elapsed,
            )

        # Защита от «галлюцинаций»: текст не должен быть сильно длиннее оригинала
        if len(result_text) > len(stripped) * self.MAX_EXPANSION:
            return EditResult(
                text=stripped, original=text,
                success=False, reason="garbled", edit_time_sec=elapsed,
            )

        return EditResult(
            text=result_text, original=text,
            success=True, reason="ok", edit_time_sec=elapsed,
        )

    # ── Внутренние методы ─────────────────────────────────────────────────────

    def _call_with_timeout(
        self, text: str, language: Optional[str]
    ) -> str:
        """Запускает LLM в отдельном потоке с таймаутом."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._llm_call, text, language)
            return future.result(timeout=self.timeout_sec)

    def _llm_call(self, text: str, language: Optional[str]) -> str:
        """Непосредственный вызов Llama через chat completion."""
        # Подсказка о языке — помогает Phi не переключаться
        lang_hint = ""
        if language:
            names = {"ru": "Russian", "en": "English"}
            lang_hint = f"\n\nLanguage of the text: {names.get(language, language)}."

        user_message = (
            f"Clean up this voice transcription text:{lang_hint}\n\n"
            f"{text}"
        )

        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system",    "content": _SYSTEM_PROMPT},
                {"role": "user",      "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["<|end|>", "<|endoftext|>", "<|user|>"],
        )

        return response["choices"][0]["message"]["content"]


# ── Консольный тест: python -m core.editor ────────────────────────────────────

# Примеры для теста (из документации, раздел 8.2)
_TEST_CASES = [
    (
        "ru",
        "ну вот значит нужно э-э сделать отчёт по проекту "
        "ну там три раздела значит введение основная часть ну и заключение",
    ),
    (
        "ru",
        "как бы я думаю ммм что нужно встретиться в понедельник "
        "ну или вот во вторник это самое в десять утра",
    ),
    (
        "en",
        "uh so basically I uh wanted to say that um the meeting "
        "is you know scheduled for like monday morning right",
    ),
]


def _load_config() -> dict:
    import yaml  # type: ignore

    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        return {
            "editor": {
                "enabled":      True,
                "model_path":   "models/Phi-3.5-mini-Q4.gguf",
                "gpu_layers":   -1,
                "context_size": 2048,
                "max_tokens":   1024,
                "temperature":  0.1,
                "timeout_sec":  15.0,
            }
        }
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Интерактивный тест Editor:
      1. Загружаем Phi-3.5 Mini
      2. Прогоняем встроенные тест-кейсы
      3. Даём возможность ввести свой текст
    """
    import sys

    config = _load_config()

    print("\n" + "═" * 64)
    print("  VoiceScribe — тест LLM постобработки (Phi-3.5 Mini)")
    print("═" * 64)
    print(f"\n  Модель: {config['editor']['model_path']}")
    print(f"  GPU слои: {config['editor']['gpu_layers']}  |  "
          f"Температура: {config['editor']['temperature']}  |  "
          f"Таймаут: {config['editor']['timeout_sec']} с\n")

    editor = Editor(config)

    if not editor.enabled:
        print("  editor.enabled = false в config.yaml. Тест пропущен.")
        return

    print("Загружаем Phi-3.5 Mini ... ", end="", flush=True)
    t0 = time.perf_counter()
    try:
        editor.load()
    except FileNotFoundError as e:
        print(f"\n\n  [ERR] {e}")
        return
    print(f"готово за {time.perf_counter() - t0:.1f} с\n")

    # ── Встроенные тест-кейсы ─────────────────────────────────────────────────
    print("─" * 64)
    print("  Встроенные тест-кейсы:")
    print("─" * 64)

    for i, (lang, raw_text) in enumerate(_TEST_CASES, 1):
        print(f"\n  [{i}] Язык: {lang.upper()}")
        print(f"  До:    {raw_text}")

        result = editor.edit(raw_text, language=lang)

        status = "✓" if result.success else f"✗ ({result.reason})"
        print(f"  После: {result.text}")
        print(f"  {status}  |  {result.edit_time_sec:.2f} с")

    # ── Ввод своего текста ────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  Введите свой текст (Enter дважды для обработки, Ctrl+C — выход):")
    print("─" * 64 + "\n")

    try:
        while True:
            lines: list[str] = []
            try:
                while True:
                    line = input()
                    if line == "" and lines:
                        break
                    lines.append(line)
            except EOFError:
                break

            if not lines:
                break

            user_text = " ".join(lines)
            print("\n  Обрабатываем ...", end="", flush=True)
            result = editor.edit(user_text)
            print(f"\r  Результат ({result.edit_time_sec:.2f} с):\n")
            print(f"  {result.text}\n")
            print("─" * 64 + "\n")

    except KeyboardInterrupt:
        pass

    print("\n  Готово.")


if __name__ == "__main__":
    main()
