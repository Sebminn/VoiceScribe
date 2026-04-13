"""
core/transcriber.py — распознавание речи через Faster-Whisper.

Параметры (из config.yaml → transcriber):
    model                  — 'large-v3-turbo' (или smaller для отладки)
    device                 — 'cuda' | 'cpu'
    compute_type           — 'int8' (~1.5 ГБ VRAM) | 'float16' (~3 ГБ)
    beam_size              — 5 (баланс точности/скорости)
    language               — null (автоопределение) | 'ru' | 'en'
    task                   — 'transcribe' | 'translate'
    condition_on_previous  — false (короткие сегменты)

Консольный тест:
    python -m core.transcriber
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── Результат транскрипции ────────────────────────────────────────────────────

@dataclass
class TranscribeResult:
    """Результат одного вызова Transcriber.transcribe()."""

    text: str                        # Полный текст (все сегменты объединены)
    language: str                    # Определённый язык ('ru', 'en', …)
    language_probability: float      # Уверенность в языке (0.0–1.0)
    audio_duration_sec: float        # Длина входного аудио (сек)
    transcribe_time_sec: float       # Время транскрипции (сек)
    segments: list[dict] = field(default_factory=list)  # [{start, end, text}]

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def realtime_factor(self) -> float:
        """audio_duration / transcribe_time; > 1 = быстрее реального времени."""
        if self.transcribe_time_sec <= 0:
            return 0.0
        return self.audio_duration_sec / self.transcribe_time_sec

    def __str__(self) -> str:
        return (
            f"[{self.language.upper()} {self.language_probability:.0%}] "
            f"{self.text}"
        )


# ── Транскрибер ───────────────────────────────────────────────────────────────

class Transcriber:
    """
    Обёртка над faster_whisper.WhisperModel.

    Модель загружается лениво: через load() при старте или автоматически
    при первом вызове transcribe().

    Приоритет пути к модели:
        1. models/faster-whisper-<name>/ (скачано setup.py)
        2. Имя модели → faster-whisper загрузит из HuggingFace
    """

    # Минимальная длина аудио для передачи в Whisper (сек)
    MIN_AUDIO_SEC: float = 0.1

    def __init__(self, config: dict) -> None:
        cfg = config["transcriber"]

        self.model_name:   str  = cfg["model"]           # large-v3-turbo
        self.device:       str  = cfg["device"]          # cuda
        self.compute_type: str  = cfg["compute_type"]    # int8
        self.beam_size:    int  = cfg.get("beam_size", 5)
        self.task:         str  = cfg.get("task", "transcribe")
        self.condition_on_previous: bool = cfg.get("condition_on_previous", False)

        # null в YAML → None в Python → автоопределение языка
        lang = cfg.get("language")
        self.language: Optional[str] = lang if lang else None

        self._model = None                 # faster_whisper.WhisperModel
        self._sample_rate: int = config["audio"]["sample_rate"]  # 16 000

    # ── Загрузка ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Загружает WhisperModel в VRAM.
        Безопасно вызывать повторно — повторная загрузка не происходит.
        Вызывайте в отдельном потоке: занимает 5–20 сек на первом запуске.
        """
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper не установлен. Запустите setup.py."
            ) from exc

        model_path = self._resolve_model_path()

        try:
            self._model = WhisperModel(
                model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            # Fallback: если CUDA недоступна — пробуем CPU с int8
            if self.device == "cuda":
                import warnings
                warnings.warn(
                    f"Не удалось загрузить модель на CUDA ({exc}). "
                    "Переключаемся на CPU/int8.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._model = WhisperModel(
                    model_path,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise

    def _resolve_model_path(self) -> str:
        """
        Возвращает путь к модели: локальный каталог (если скачан setup.py)
        или имя модели для авто-загрузки из HuggingFace.
        """
        root = Path(__file__).parent.parent
        local_dir = root / "models" / f"faster-whisper-{self.model_name}"
        if local_dir.exists() and any(local_dir.iterdir()):
            return str(local_dir)
        return self.model_name  # faster-whisper скачает сам

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    # ── Транскрипция ──────────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray) -> TranscribeResult:
        """
        Транскрибирует аудио-массив (float32, 16kHz, моно).

        Аргументы:
            audio — np.ndarray shape (N,), float32, значения в [-1, 1]

        Возвращает TranscribeResult.
        Если аудио слишком короткое — возвращает пустой результат немедленно.
        """
        self._ensure_loaded()

        audio_f32 = audio.astype(np.float32)
        audio_duration = len(audio_f32) / self._sample_rate

        if audio_duration < self.MIN_AUDIO_SEC:
            return TranscribeResult(
                text="",
                language=self.language or "unknown",
                language_probability=0.0,
                audio_duration_sec=audio_duration,
                transcribe_time_sec=0.0,
            )

        t_start = time.perf_counter()

        segments_gen, info = self._model.transcribe(
            audio_f32,
            beam_size=self.beam_size,
            language=self.language,
            task=self.task,
            condition_on_previous_text=self.condition_on_previous,
            vad_filter=False,   # используем Silero VAD из core/vad.py
            word_timestamps=False,
        )

        # Генератор — материализуем здесь, чтобы замерить время честно
        segments: list[dict] = []
        text_parts: list[str] = []
        for seg in segments_gen:
            seg_text = seg.text.strip()
            if seg_text:
                text_parts.append(seg_text)
                segments.append(
                    {"start": round(seg.start, 2),
                     "end":   round(seg.end,   2),
                     "text":  seg_text}
                )

        transcribe_time = time.perf_counter() - t_start
        full_text = " ".join(text_parts)

        return TranscribeResult(
            text=full_text,
            language=info.language,
            language_probability=float(info.language_probability),
            audio_duration_sec=round(audio_duration, 2),
            transcribe_time_sec=round(transcribe_time, 3),
            segments=segments,
        )

    def transcribe_file(self, path: str | Path) -> TranscribeResult:
        """
        Транскрибирует WAV/MP3/… файл.
        Удобно для консольного тестирования.
        """
        import soundfile as sf  # type: ignore (опциональная зависимость)

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if sr != self._sample_rate:
            # Простейший ресэмплинг через scipy, если частоты не совпадают
            try:
                from scipy.signal import resample_poly  # type: ignore
                import math
                gcd = math.gcd(self._sample_rate, sr)
                data = resample_poly(data, self._sample_rate // gcd, sr // gcd)
            except ImportError:
                raise RuntimeError(
                    f"Файл {path} имеет частоту {sr} Гц, "
                    f"требуется {self._sample_rate} Гц. "
                    "Установите scipy для авто-ресэмплинга."
                )
        # Если стерео — берём первый канал
        if data.ndim > 1:
            data = data[:, 0]
        return self.transcribe(data)


# ── Консольный тест: python -m core.transcriber ───────────────────────────────

def _load_config() -> dict:
    import yaml  # type: ignore

    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        return {
            "audio":       {"device": "default", "sample_rate": 16000,
                            "channels": 1, "chunk_ms": 32},
            "vad":         {"threshold": 0.5, "min_speech_ms": 250,
                            "min_silence_ms": 700, "pre_speech_pad_ms": 300},
            "transcriber": {"model": "large-v3-turbo", "device": "cuda",
                            "compute_type": "int8", "beam_size": 5,
                            "language": None, "task": "transcribe",
                            "condition_on_previous": False},
        }
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Интерактивный тест:
      1. Загружаем модель
      2. Записываем аудио (удержание Enter или задержка 5 сек)
      3. Показываем результат транскрипции + метрики
    """
    import threading

    config = _load_config()

    print("\n" + "═" * 62)
    print("  VoiceScribe — тест Faster-Whisper STT")
    print("═" * 62)

    # ── Загрузка модели ───────────────────────────────────────────────────────
    transcriber = Transcriber(config)
    print(f"\nМодель : {config['transcriber']['model']}")
    print(f"Устройство: {config['transcriber']['device']} / "
          f"{config['transcriber']['compute_type']}")
    print("\nЗагружаем Faster-Whisper ... ", end="", flush=True)
    t0 = time.perf_counter()
    transcriber.load()
    print(f"готово за {time.perf_counter() - t0:.1f} с\n")

    # ── Загрузка VAD (опционально) ────────────────────────────────────────────
    vad = None
    try:
        from core.vad import VAD  # noqa: PLC0415
        print("Загружаем Silero VAD ... ", end="", flush=True)
        vad = VAD(config)
        vad.load()
        print("готов.\n")
    except Exception as exc:
        print(f"VAD недоступен ({exc}). Запись без VAD.\n")

    # ── Запись аудио ──────────────────────────────────────────────────────────
    from core.audio import AudioCapture  # noqa: PLC0415

    cap = AudioCapture(config)

    print("─" * 62)
    print("  Нажмите ENTER чтобы начать запись.")
    print("  Нажмите ENTER снова чтобы остановить (или Ctrl+C).")
    print("─" * 62)
    input()

    cap.start()
    all_chunks: list[np.ndarray] = []
    stop_event = threading.Event()

    def _record() -> None:
        while not stop_event.is_set():
            chunk = cap.read_chunk(timeout=0.1)
            if chunk is not None:
                all_chunks.append(chunk)

    record_thread = threading.Thread(target=_record, daemon=True)
    record_thread.start()

    start_time = time.time()
    print("\n  ◉ Запись идёт ... нажмите ENTER для остановки\n")

    # Таймер в отдельном потоке
    def _timer() -> None:
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r  ⏱  {elapsed:.1f} с          ")
            sys.stdout.flush()
            time.sleep(0.1)

    timer_thread = threading.Thread(target=_timer, daemon=True)
    timer_thread.start()

    try:
        input()
    except KeyboardInterrupt:
        pass

    stop_event.set()
    cap.stop()
    record_thread.join(timeout=1)
    elapsed_total = time.time() - start_time
    print(f"\n\n  Записано: {elapsed_total:.1f} с")

    if not all_chunks:
        print("  Нет аудиоданных. Выход.")
        return

    audio = np.concatenate(all_chunks)

    # ── VAD: обрезаем тишину ──────────────────────────────────────────────────
    if vad is not None:
        audio_trimmed = vad.trim_silence(audio)
        trimmed_sec = len(audio_trimmed) / config["audio"]["sample_rate"]
        original_sec = len(audio) / config["audio"]["sample_rate"]
        if len(audio_trimmed) < len(audio):
            print(f"  VAD: {original_sec:.1f} с → {trimmed_sec:.1f} с "
                  f"(убрано {original_sec - trimmed_sec:.1f} с тишины)")
        if not vad.has_speech(audio):
            print("  VAD: речь не обнаружена. Выход.")
            return
        audio = audio_trimmed

    # ── Транскрипция ──────────────────────────────────────────────────────────
    print("\n  Транскрибируем ...", end="", flush=True)
    result = transcriber.transcribe(audio)
    print(" готово.\n")

    print("─" * 62)
    if result.is_empty:
        print("  (текст не распознан)")
    else:
        print(f"  {result}")
    print("─" * 62)
    print(f"\n  Аудио:       {result.audio_duration_sec:.2f} с")
    print(f"  Время STT:   {result.transcribe_time_sec:.3f} с")
    print(f"  RTF:         {result.realtime_factor:.1f}x  "
          f"({'быстрее' if result.realtime_factor >= 1 else 'медленнее'} реального времени)")
    if result.segments:
        print(f"\n  Сегменты ({len(result.segments)}):")
        for s in result.segments:
            print(f"    [{s['start']:5.2f}–{s['end']:5.2f}] {s['text']}")
    print()


if __name__ == "__main__":
    main()
