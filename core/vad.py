"""
core/vad.py — детекция речи через Silero VAD v4 (CPU).

Параметры (из config.yaml → vad):
    threshold         — порог вероятности речи (0.0–1.0)
    min_speech_ms     — минимальная длина речевого сегмента (мс)
    min_silence_ms    — длина паузы для завершения сегмента (мс)
    pre_speech_pad_ms — сколько аудио захватываем до начала речи (мс)

Два режима работы:
    1. Потоковый (pipeline, toggle-режим): process_chunk() / reset()
    2. Пакетный  (hold-режим, постобработка): trim_silence() / has_speech()

Консольный тест:
    python -m core.vad
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


class VAD:
    """
    Обёртка над Silero VAD v4.

    Модель загружается лениво: явно через load() или автоматически
    при первом обращении к is_speech() / trim_silence() / process_chunk().
    """

    # Silero VAD v4 требует окно ровно 512 сэмплов @ 16 000 Гц
    WINDOW_SAMPLES: int = 512

    def __init__(self, config: dict) -> None:
        vad_cfg   = config["vad"]
        audio_cfg = config["audio"]

        self.threshold:         float = vad_cfg["threshold"]          # 0.5
        self.min_speech_ms:     int   = vad_cfg["min_speech_ms"]      # 250
        self.min_silence_ms:    int   = vad_cfg["min_silence_ms"]     # 700
        self.pre_speech_pad_ms: int   = vad_cfg["pre_speech_pad_ms"]  # 300
        self.sample_rate:       int   = audio_cfg["sample_rate"]      # 16 000

        # Производные в сэмплах
        self._min_speech_samples:    int = int(self.sample_rate * self.min_speech_ms / 1000)
        self._min_silence_samples:   int = int(self.sample_rate * self.min_silence_ms / 1000)
        self._pre_speech_pad_samples: int = int(self.sample_rate * self.pre_speech_pad_ms / 1000)

        # Модель и итератор — загружаются в load()
        self._model = None
        self._vad_iterator = None

        # ── Состояние потокового VAD ──────────────────────────────────────────
        self._speech_active:   bool           = False
        self._silence_counter: int            = 0    # сэмплов тишины подряд
        self._speech_counter:  int            = 0    # сэмплов речи подряд
        # Кольцевой буфер: хранит pre-speech-pad последних сэмплов
        self._pre_buffer:      list[np.ndarray] = []
        self._pre_buffer_len:  int              = 0
        # Накопленный речевой сегмент
        self._segment_chunks:  list[np.ndarray] = []
        self._segment_samples: int              = 0
        # Внутренний буфер для выравнивания чанков под WINDOW_SAMPLES
        self._window_buf: np.ndarray = np.empty(0, dtype=np.float32)

    # ── Загрузка модели ───────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Загружает Silero VAD v4 и создаёт VADIterator.
        Безопасно вызывать повторно — повторная загрузка не происходит.
        """
        if self._model is not None:
            return

        try:
            from silero_vad import VADIterator, load_silero_vad  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "silero-vad не установлен. Запустите: pip install silero-vad"
            ) from exc

        self._model = load_silero_vad(onnx=False)

        # VADIterator используется только для потокового режима;
        # параметры выравниваем с нашим конфигом.
        self._vad_iterator = VADIterator(
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.pre_speech_pad_ms,
        )

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    # ── Потоковый VAD ─────────────────────────────────────────────────────────

    def is_speech(self, chunk: np.ndarray) -> bool:
        """
        Вероятностная детекция речи в одном чанке.
        Чанк должен содержать ровно WINDOW_SAMPLES (512) сэмплов float32.
        Если размер другой — буферизуем и возвращаем последний результат.

        Возвращает True, если речь обнаружена (вероятность > threshold).
        """
        self._ensure_loaded()

        import torch  # type: ignore

        # Если чанк ровно 512 — обрабатываем напрямую
        if len(chunk) == self.WINDOW_SAMPLES:
            t = torch.from_numpy(chunk)
            prob: float = self._model(t, self.sample_rate).item()
            return prob >= self.threshold

        # Иначе: дополняем до 512 нулями или берём среднее нескольких окон
        padded = np.zeros(self.WINDOW_SAMPLES, dtype=np.float32)
        n = min(len(chunk), self.WINDOW_SAMPLES)
        padded[:n] = chunk[:n]
        t = torch.from_numpy(padded)
        prob = self._model(t, self.sample_rate).item()
        return prob >= self.threshold

    def process_chunk(self, chunk: np.ndarray) -> tuple[bool, Optional[np.ndarray]]:
        """
        Потоковая обработка одного чанка (для toggle-режима и авто-стопа).

        Возвращает:
            (speech_active, segment)
            - speech_active: True, если сейчас идёт речь
            - segment: готовый речевой сегмент (ndarray) когда речь завершилась,
                       иначе None

        Логика:
            SILENCE → SPEECH: вероятность ≥ threshold на min_speech_samples
            SPEECH → SILENCE: тишина ≥ min_silence_samples

        Готовый сегмент включает pre_speech_pad спереди.
        """
        self._ensure_loaded()

        import torch  # type: ignore

        # Добавляем чанк во внутренний буфер для выравнивания
        self._window_buf = np.concatenate([self._window_buf, chunk])
        completed_segment: Optional[np.ndarray] = None

        # Обрабатываем все полные окна по 512 сэмплов
        while len(self._window_buf) >= self.WINDOW_SAMPLES:
            window = self._window_buf[: self.WINDOW_SAMPLES]
            self._window_buf = self._window_buf[self.WINDOW_SAMPLES :]

            t = torch.from_numpy(window)
            prob: float = self._model(t, self.sample_rate).item()
            is_speech_window = prob >= self.threshold

            if not self._speech_active:
                # ── Режим ТИШИНА ─────────────────────────────────────────────
                # Поддерживаем pre-speech-pad буфер
                self._pre_buffer.append(window)
                self._pre_buffer_len += len(window)
                # Удаляем старые чанки сверх pre_speech_pad
                while self._pre_buffer_len > self._pre_speech_pad_samples:
                    removed = self._pre_buffer.pop(0)
                    self._pre_buffer_len -= len(removed)

                if is_speech_window:
                    self._speech_counter += len(window)
                    if self._speech_counter >= self._min_speech_samples:
                        # Переходим в РЕЧЬ: сбрасываем данные pre-pad в сегмент
                        self._speech_active = True
                        self._silence_counter = 0
                        self._speech_counter = 0
                        # Переносим pre-pad буфер в сегмент
                        self._segment_chunks = list(self._pre_buffer)
                        self._segment_samples = self._pre_buffer_len
                        self._pre_buffer = []
                        self._pre_buffer_len = 0
                else:
                    self._speech_counter = 0

            else:
                # ── Режим РЕЧЬ ────────────────────────────────────────────────
                self._segment_chunks.append(window)
                self._segment_samples += len(window)

                if is_speech_window:
                    self._silence_counter = 0
                else:
                    self._silence_counter += len(window)
                    if self._silence_counter >= self._min_silence_samples:
                        # Речь завершилась → собираем сегмент
                        self._speech_active = False
                        completed_segment = np.concatenate(self._segment_chunks)
                        # Обрезаем trailing-тишину (оставляем post-pad = pre_speech_pad_ms)
                        trim_len = max(
                            0,
                            self._segment_samples
                            - self._silence_counter
                            + self._pre_speech_pad_samples,
                        )
                        if trim_len < len(completed_segment):
                            completed_segment = completed_segment[:trim_len]
                        # Сброс
                        self._segment_chunks = []
                        self._segment_samples = 0
                        self._silence_counter = 0
                        self._pre_buffer = []
                        self._pre_buffer_len = 0

        return self._speech_active, completed_segment

    def get_current_segment(self) -> Optional[np.ndarray]:
        """
        Возвращает накопленный на данный момент сегмент (без ожидания тишины).
        Используется при принудительной остановке (hold-режим: key release).
        """
        if not self._segment_chunks:
            return None
        return np.concatenate(self._segment_chunks)

    def reset(self) -> None:
        """Сбрасывает потоковое состояние (после каждой сессии записи)."""
        self._speech_active = False
        self._silence_counter = 0
        self._speech_counter = 0
        self._pre_buffer = []
        self._pre_buffer_len = 0
        self._segment_chunks = []
        self._segment_samples = 0
        self._window_buf = np.empty(0, dtype=np.float32)
        if self._vad_iterator is not None:
            self._vad_iterator.reset_states()

    # ── Пакетный VAD (для hold-режима) ───────────────────────────────────────

    def has_speech(self, audio: np.ndarray) -> bool:
        """
        Проверяет, содержит ли массив audio хоть один речевой сегмент.
        Используется для валидации записи в hold-режиме.
        """
        self._ensure_loaded()
        import torch  # type: ignore
        from silero_vad import get_speech_timestamps  # type: ignore

        tensor = torch.from_numpy(audio.astype(np.float32))
        timestamps = get_speech_timestamps(
            tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_ms,
        )
        return len(timestamps) > 0

    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Обрезает начальную и конечную тишину из аудио (hold-режим).
        Добавляет pre_speech_pad спереди и сзади, если места достаточно.

        Если речь не найдена — возвращает исходный массив без изменений.
        """
        self._ensure_loaded()
        import torch  # type: ignore
        from silero_vad import get_speech_timestamps  # type: ignore

        audio_f32 = audio.astype(np.float32)
        tensor = torch.from_numpy(audio_f32)

        timestamps = get_speech_timestamps(
            tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_ms,
        )

        if not timestamps:
            return audio_f32  # речи нет — возвращаем как есть

        pad = self._pre_speech_pad_samples
        start = max(0, timestamps[0]["start"] - pad)
        end   = min(len(audio_f32), timestamps[-1]["end"] + pad)
        return audio_f32[start:end]


# ── Консольный тест: python -m core.vad ───────────────────────────────────────

def _load_config() -> dict:
    import yaml  # type: ignore

    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        return {
            "audio": {"device": "default", "sample_rate": 16000,
                      "channels": 1, "chunk_ms": 32},
            "vad":   {"threshold": 0.5, "min_speech_ms": 250,
                      "min_silence_ms": 700, "pre_speech_pad_ms": 300},
        }
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Реал-тайм тест VAD: показывает события SPEECH START / END."""
    config = _load_config()

    print("\n═" * 60)
    print("  VoiceScribe — тест Silero VAD v4")
    print("═" * 60)
    print(f"\n  Порог: {config['vad']['threshold']}  |  "
          f"Мин. речь: {config['vad']['min_speech_ms']} мс  |  "
          f"Мин. тишина: {config['vad']['min_silence_ms']} мс\n")

    print("Загружаем Silero VAD ... ", end="", flush=True)
    vad = VAD(config)
    vad.load()
    print("готов.\n")

    from core.audio import AudioCapture  # noqa: PLC0415

    cap = AudioCapture(config)
    cap.start()
    print("Говорите. Ctrl+C — выход.\n")

    segment_count = 0
    speech_start: Optional[float] = None

    try:
        while True:
            chunk = cap.read_chunk(timeout=0.1)
            if chunk is None:
                continue

            was_active = vad._speech_active
            is_active, segment = vad.process_chunk(chunk)

            # Начало речи
            if is_active and not was_active:
                speech_start = time.time()
                print(f"  ◉ РЕЧЬ начата")

            # Конец речи — получили готовый сегмент
            if segment is not None:
                segment_count += 1
                duration_ms = len(segment) / config["audio"]["sample_rate"] * 1000
                elapsed = (
                    f"{time.time() - speech_start:.1f} с"
                    if speech_start else "?"
                )
                print(f"  ○ тишина — сегмент #{segment_count}: "
                      f"{duration_ms:.0f} мс аудио ({elapsed})")
                speech_start = None

            # Индикатор уровня
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            filled = min(int(rms / 0.15 * 25), 25)
            bar = "█" * filled + "░" * (25 - filled)
            marker = "◉" if is_active else "○"
            sys.stdout.write(f"\r  {marker} {bar}  {rms:.4f}          ")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nОстановлено.")
    finally:
        cap.stop()
        print(f"Всего сегментов: {segment_count}")


if __name__ == "__main__":
    main()
