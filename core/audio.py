"""
core/audio.py — захват PCM-аудио с микрофона.

Параметры (из config.yaml → audio):
    device      — 'default' или числовой индекс устройства
    sample_rate — 16 000 Гц (НЕ МЕНЯТЬ: требование Faster-Whisper)
    channels    — 1, моно (НЕ МЕНЯТЬ)
    chunk_ms    — 32 мс = 512 сэмплов; совпадает с окном Silero VAD v4

Консольный тест:
    python -m core.audio
"""

from __future__ import annotations

import queue
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioCapture:
    """
    Захватывает PCM 16kHz моно из микрофона и раздаёт чанки через очередь.

    Использование:
        cap = AudioCapture(config)
        cap.start()
        while recording:
            chunk = cap.read_chunk()   # np.ndarray float32, 512 сэмплов
        cap.stop()
    """

    def __init__(self, config: dict) -> None:
        audio_cfg = config["audio"]

        self.sample_rate: int = audio_cfg["sample_rate"]  # 16 000
        self.channels: int    = audio_cfg["channels"]     # 1
        self.chunk_ms: int    = audio_cfg["chunk_ms"]     # 32

        device_cfg  = audio_cfg.get("device", "default")
        self.device = None if device_cfg == "default" else device_cfg

        # 32 мс * 16 000 Гц / 1000 = 512 сэмплов
        self.chunk_size: int = int(self.sample_rate * self.chunk_ms / 1000)

        # Очередь чанков; 200 * 32 мс ≈ 6.4 с буфера до переполнения
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: Optional[sd.InputStream] = None
        self._running: bool = False

    # ── Управление потоком ────────────────────────────────────────────────────

    def start(self) -> None:
        """Открывает аудиопоток и начинает захват."""
        if self._running:
            return
        # Очищаем буфер от старых данных
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        """Останавливает захват и освобождает устройство."""
        if not self._running:
            return
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Чтение данных ─────────────────────────────────────────────────────────

    def read_chunk(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """
        Возвращает следующий чанк PCM (float32, моно, chunk_size сэмплов).
        Блокирует до timeout секунд; возвращает None при таймауте.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_all(self) -> np.ndarray:
        """
        Считывает все накопленные в очереди чанки и возвращает их
        конкатенацией. Не блокирует.
        """
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks)

    # ── Информация об устройствах ─────────────────────────────────────────────

    @staticmethod
    def list_devices() -> list[dict]:
        """Возвращает список устройств ввода (микрофонов)."""
        devices = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": d["name"],
                        "channels": d["max_input_channels"],
                        "default_sr": int(d["default_samplerate"]),
                    }
                )
        return devices

    @staticmethod
    def default_input_device() -> dict:
        """Возвращает имя и индекс системного устройства ввода по умолчанию."""
        idx = sd.default.device[0]
        if idx < 0:
            return {"index": -1, "name": "(не определено)"}
        d = sd.query_devices(idx)
        return {"index": idx, "name": d["name"]}

    # ── Вспомогательное ───────────────────────────────────────────────────────

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice-callback: вызывается в потоке захвука."""
        # status.input_overflow → пропущенные сэмплы — молча игнорируем
        chunk = indata[:, 0].copy()  # берём только первый канал (моно)
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # Буфер переполнен: выбрасываем старый чанк, кладём новый
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass


# ── Консольный тест: python -m core.audio ─────────────────────────────────────

def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk ** 2)))


def _bar(rms: float, width: int = 30) -> str:
    """Горизонтальный ASCII-индикатор уровня громкости."""
    # RMS float32 микрофона обычно в диапазоне 0.0–0.3; нормируем на 0.15
    filled = min(int(rms / 0.15 * width), width)
    return "█" * filled + "░" * (width - filled)


def _load_config() -> dict:
    """Загружает config.yaml из корня проекта."""
    import yaml

    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.exists():
        # Минимальные дефолты для автономного теста
        return {
            "audio": {
                "device": "default",
                "sample_rate": 16000,
                "channels": 1,
                "chunk_ms": 32,
            },
            "vad": {
                "threshold": 0.5,
                "min_speech_ms": 250,
                "min_silence_ms": 700,
                "pre_speech_pad_ms": 300,
            },
        }
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Интерактивный тест захвата аудио + VAD в реальном времени."""
    config = _load_config()

    print("\n═" * 60)
    print("  VoiceScribe — тест аудио + VAD")
    print("═" * 60)

    # Список устройств
    print("\nДоступные устройства ввода:")
    for d in AudioCapture.list_devices():
        print(f"  [{d['index']:2d}] {d['name']}  ({d['channels']} ch, {d['default_sr']} Hz)")

    default = AudioCapture.default_input_device()
    print(f"\nТекущее по умолчанию: [{default['index']}] {default['name']}")
    print(f"Параметры: {config['audio']['sample_rate']} Гц, "
          f"моно, чанки {config['audio']['chunk_ms']} мс\n")

    # Попытка загрузить VAD (необязательно для теста аудио)
    vad = None
    try:
        from core.vad import VAD  # noqa: PLC0415
        print("Загружаем Silero VAD ... ", end="", flush=True)
        vad = VAD(config)
        vad.load()
        print("готов.\n")
    except Exception as exc:
        print(f"VAD недоступен ({exc}). Показываем только уровень сигнала.\n")

    cap = AudioCapture(config)
    cap.start()

    print("Говорите в микрофон. Ctrl+C — выход.\n")
    speech_was_active = False

    try:
        while True:
            chunk = cap.read_chunk(timeout=0.1)
            if chunk is None:
                continue

            rms = _rms(chunk)
            bar = _bar(rms)

            # VAD
            speech_label = ""
            if vad is not None:
                is_speech = vad.is_speech(chunk)
                if is_speech and not speech_was_active:
                    speech_label = "  ◉ РЕЧЬ"
                    speech_was_active = True
                elif not is_speech and speech_was_active:
                    speech_label = "  ○ тишина"
                    speech_was_active = False
                elif is_speech:
                    speech_label = "  ◉ РЕЧЬ"
                # else: тишина → пустая метка

            # Перезаписываем строку в терминале
            line = f"\r  {bar}  {rms:.4f}{speech_label}          "
            sys.stdout.write(line)
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nОстановлено.")
    finally:
        cap.stop()
        print("Готово.")


if __name__ == "__main__":
    main()
