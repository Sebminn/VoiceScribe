"""
core/pipeline.py — оркестратор пайплайна VoiceScribe.

Полный путь: микрофон → VAD → Faster-Whisper → Phi-3.5 Mini → буфер обмена.

Состояния:
    IDLE → RECORDING → PROCESSING → DONE → IDLE
                                  ↘ ERROR → IDLE

Два режима записи (из config.yaml → hotkey.mode):
    hold   — запись пока клавиша удержана; при отпускании → обработка
    toggle — первое нажатие = старт; второе = стоп; или авто-стоп по тишине

Callbacks (устанавливаются снаружи, до start_recording):
    on_state_change(state)          — смена состояния (для иконки трея)
    on_result(text, result)         — текст готов и скопирован в буфер
    on_error(message)               — ошибка обработки
    on_recording_tick(elapsed_sec)  — тик каждые ~0.5 с (для таймера трея)

Консольный тест:
    python -m core.pipeline
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# ── Состояния пайплайна ───────────────────────────────────────────────────────

class PipelineState(Enum):
    IDLE       = auto()   # Ожидание; горячая клавиша активна
    RECORDING  = auto()   # Идёт запись; иконка красная
    PROCESSING = auto()   # Whisper + Phi обрабатывают; иконка жёлтая
    DONE       = auto()   # Текст скопирован; иконка зелёная (2 сек)
    ERROR      = auto()   # Ошибка; иконка красная+! (кликнуть для деталей)


# ── Результат пайплайна ───────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Полная информация о завершённой сессии записи."""

    text:               str    # Итоговый текст в буфере обмена
    language:           str    # Определённый язык ('ru', 'en', …)
    audio_duration_sec: float  # Длина речи после VAD-trim (сек)
    transcribe_sec:     float  # Время Faster-Whisper (сек)
    edit_sec:           float  # Время Phi-3.5 Mini (сек; 0 если disabled)
    total_sec:          float  # Полное время от stop до clipboard
    llm_used:           bool   # True = LLM обработал без fallback
    segments:           list   = field(default_factory=list)

    def summary(self) -> str:
        rtf = (self.audio_duration_sec / self.transcribe_sec
               if self.transcribe_sec > 0 else 0)
        llm_tag = "LLM✓" if self.llm_used else "LLM✗"
        return (
            f"[{self.language.upper()} | STT {self.transcribe_sec:.2f}s "
            f"RTF×{rtf:.1f} | {llm_tag} {self.edit_sec:.2f}s | "
            f"Σ {self.total_sec:.2f}s]"
        )


# ── Пайплайн ──────────────────────────────────────────────────────────────────

class Pipeline:
    """
    Центральный оркестратор VoiceScribe.

    Использование:
        pipeline = Pipeline(config)
        pipeline.on_state_change = lambda s: ...
        pipeline.on_result       = lambda text, r: ...
        pipeline.on_error        = lambda msg: ...
        pipeline.load_models()          # один раз при старте
        pipeline.start_recording()      # при нажатии клавиши
        pipeline.stop_recording()       # при отпускании (hold) / повторном нажатии (toggle)
    """

    def __init__(self, config: dict) -> None:
        from core.audio       import AudioCapture   # noqa: PLC0415
        from core.editor      import Editor         # noqa: PLC0415
        from core.transcriber import Transcriber    # noqa: PLC0415
        from core.vad         import VAD            # noqa: PLC0415

        self._config = config
        hotkey_cfg   = config.get("hotkey", {})
        audio_cfg    = config.get("audio",  {})

        self.mode:             str   = hotkey_cfg.get("mode",            "hold")
        self.min_hold_ms:      int   = hotkey_cfg.get("min_hold_ms",      500)
        self.autostop_sec:     float = hotkey_cfg.get("autostop_sec",     30.0)
        self.max_record_sec:   float = hotkey_cfg.get("max_record_sec",   300.0)
        self._sample_rate:     int   = audio_cfg.get("sample_rate",       16000)

        output_cfg = config.get("output", {})
        self._notify:          bool  = output_cfg.get("notify",            True)
        self._preview_chars:   int   = output_cfg.get("notify_preview_chars", 60)

        # Компоненты
        self.audio       = AudioCapture(config)
        self.vad         = VAD(config)
        self.transcriber = Transcriber(config)
        self.editor      = Editor(config)

        # Callbacks
        self.on_state_change:   Optional[Callable[[PipelineState], None]]      = None
        self.on_result:         Optional[Callable[[str, PipelineResult], None]] = None
        self.on_error:          Optional[Callable[[str], None]]                 = None
        self.on_recording_tick: Optional[Callable[[float], None]]               = None

        # Внутреннее состояние
        self._state      = PipelineState.IDLE
        self._state_lock = threading.Lock()

        self._stop_event   = threading.Event()
        self._record_chunks: list[np.ndarray] = []
        self._record_start: float = 0.0
        self._last_error:   str   = ""

        self._record_thread:  Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

    # ── Загрузка моделей ──────────────────────────────────────────────────────

    def load_models(
        self,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        """
        Загружает VAD, Faster-Whisper и Phi-3.5 Mini.
        Вызывать один раз при старте в отдельном потоке (занимает 10–30 сек).

        on_progress(message, fraction 0.0–1.0) — опциональный callback прогресса.
        """
        def _progress(msg: str, frac: float) -> None:
            if on_progress:
                on_progress(msg, frac)

        _progress("Загрузка Silero VAD...", 0.1)
        self.vad.load()

        _progress("Загрузка Faster-Whisper...", 0.4)
        self.transcriber.load()

        if self.editor.enabled:
            _progress("Загрузка Phi-3.5 Mini...", 0.75)
            self.editor.load()

        _progress("Готово!", 1.0)

    # ── Управление записью ────────────────────────────────────────────────────

    def start_recording(self) -> bool:
        """
        Начинает запись.
        Thread-safe. Возвращает False если уже не в IDLE (защита от двойного нажатия).
        """
        with self._state_lock:
            if self._state != PipelineState.IDLE:
                return False
            self._set_state_locked(PipelineState.RECORDING)

        self._stop_event.clear()
        self._record_chunks = []
        self.vad.reset()
        self._record_start = time.time()

        self.audio.start()

        self._record_thread = threading.Thread(
            target=self._record_loop, daemon=True, name="vs-record"
        )
        self._record_thread.start()
        return True

    def stop_recording(self) -> None:
        """
        Останавливает запись и запускает обработку.
        Thread-safe. Вызывать при отпускании клавиши (hold) или повторном нажатии (toggle).
        """
        with self._state_lock:
            if self._state != PipelineState.RECORDING:
                return
        self._stop_event.set()

    def toggle_recording(self) -> None:
        """
        Удобный метод для toggle-режима:
        IDLE → start_recording(); RECORDING → stop_recording().
        """
        with self._state_lock:
            state = self._state
        if state == PipelineState.IDLE:
            self.start_recording()
        elif state == PipelineState.RECORDING:
            self.stop_recording()

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def last_error(self) -> str:
        return self._last_error

    # ── Поток записи ─────────────────────────────────────────────────────────

    def _record_loop(self) -> None:
        """
        Фоновый поток: читает чанки из AudioCapture, мониторит VAD.
        Завершается при: stop_event, авто-стоп (toggle), max_record_sec.
        После завершения запускает _process_audio().
        """
        had_speech         = False
        last_speech_time   = time.time()
        last_tick_time     = time.time()

        while not self._stop_event.is_set():
            chunk = self.audio.read_chunk(timeout=0.1)
            if chunk is None:
                # Проверяем таймаут даже без нового чанка
                elapsed = time.time() - self._record_start
                if elapsed > self.max_record_sec:
                    break
                continue

            self._record_chunks.append(chunk)
            elapsed = time.time() - self._record_start

            # Тик для UI (раз в ~0.5 сек)
            if time.time() - last_tick_time >= 0.5:
                last_tick_time = time.time()
                if self.on_recording_tick:
                    self.on_recording_tick(elapsed)

            # Лимит длины записи
            if elapsed > self.max_record_sec:
                break

            # Авто-стоп по тишине (только в toggle-режиме)
            if self.mode == "toggle":
                is_speech = self.vad.is_speech(chunk)
                if is_speech:
                    had_speech = True
                    last_speech_time = time.time()
                elif had_speech:
                    silence_duration = time.time() - last_speech_time
                    if silence_duration >= self.autostop_sec:
                        break

        self.audio.stop()

        # Передаём управление потоку обработки
        audio_data = (
            np.concatenate(self._record_chunks)
            if self._record_chunks
            else np.array([], dtype=np.float32)
        )
        record_duration = time.time() - self._record_start

        self._process_thread = threading.Thread(
            target=self._process_audio,
            args=(audio_data, record_duration),
            daemon=True,
            name="vs-process",
        )
        self._process_thread.start()

    # ── Поток обработки ───────────────────────────────────────────────────────

    def _process_audio(self, audio: np.ndarray, record_duration: float) -> None:
        """
        Фоновый поток: VAD trim → transcribe → edit → clipboard → callback.
        """
        t_total_start = time.perf_counter()

        # ── Валидация длительности (hold-режим) ──────────────────────────────
        if self.mode == "hold":
            if record_duration * 1000 < self.min_hold_ms:
                self._finish_error(
                    f"Запись слишком короткая ({record_duration * 1000:.0f} мс < "
                    f"{self.min_hold_ms} мс). Удерживайте клавишу дольше."
                )
                return

        # ── Пустой буфер ─────────────────────────────────────────────────────
        if len(audio) == 0:
            self._finish_error("Нет аудиоданных.")
            return

        self._set_state(PipelineState.PROCESSING)

        # ── VAD: обрезаем тишину и проверяем наличие речи ────────────────────
        audio_trimmed = self.vad.trim_silence(audio)
        if not self.vad.has_speech(audio_trimmed):
            self._finish_error("Речь не обнаружена.")
            return

        # ── Faster-Whisper STT ────────────────────────────────────────────────
        try:
            stt_result = self.transcriber.transcribe(audio_trimmed)
        except Exception as exc:  # noqa: BLE001
            self._finish_error(f"Ошибка транскрипции: {exc}")
            return

        if stt_result.is_empty:
            self._finish_error("Whisper не распознал текст.")
            return

        # ── Phi-3.5 Mini постобработка ────────────────────────────────────────
        try:
            edit_result = self.editor.edit(
                stt_result.text, language=stt_result.language
            )
        except Exception as exc:  # noqa: BLE001
            # Редактор упал — используем сырой текст
            from core.editor import EditResult  # noqa: PLC0415
            edit_result = EditResult(
                text=stt_result.text, original=stt_result.text,
                success=False, reason=f"exception: {exc}", edit_time_sec=0.0,
            )

        final_text = edit_result.text
        total_sec  = time.perf_counter() - t_total_start

        # ── Копируем в буфер обмена ───────────────────────────────────────────
        try:
            import pyperclip  # type: ignore
            pyperclip.copy(final_text)
        except Exception as exc:  # noqa: BLE001
            self._finish_error(f"Ошибка буфера обмена: {exc}")
            return

        # ── Автовставка Ctrl+V (Windows SendInput) ───────────────────────────
        if self._config.get("output", {}).get("auto_paste", True):
            try:
                import ctypes
                import ctypes.wintypes as wintypes
                import time as _time

                _time.sleep(0.25)  # ждём, пока ОС обработает отпускание горячей клавиши

                INPUT_KEYBOARD   = 1
                KEYEVENTF_KEYUP  = 0x0002
                VK_CONTROL       = 0x11
                VK_V             = 0x56

                class _KEYBDINPUT(ctypes.Structure):
                    _fields_ = [
                        ("wVk",         wintypes.WORD),
                        ("wScan",       wintypes.WORD),
                        ("dwFlags",     wintypes.DWORD),
                        ("time",        wintypes.DWORD),
                        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
                    ]

                class _INPUT(ctypes.Structure):
                    class _U(ctypes.Union):
                        _fields_ = [
                            ("ki",   _KEYBDINPUT),
                            # MOUSEINPUT — самый большой член union (32 байта на 64-bit);
                            # без него sizeof(INPUT) = 32 вместо 40, и SendInput молча падает
                            ("_pad", ctypes.c_byte * 32),
                        ]
                    _anonymous_ = ("_u",)
                    _fields_    = [("type", wintypes.DWORD), ("_u", _U)]

                inputs = (_INPUT * 4)()

                inputs[0].type    = INPUT_KEYBOARD   # нажать Ctrl
                inputs[0].ki.wVk  = VK_CONTROL

                inputs[1].type    = INPUT_KEYBOARD   # нажать V
                inputs[1].ki.wVk  = VK_V

                inputs[2].type         = INPUT_KEYBOARD   # отпустить V
                inputs[2].ki.wVk       = VK_V
                inputs[2].ki.dwFlags   = KEYEVENTF_KEYUP

                inputs[3].type         = INPUT_KEYBOARD   # отпустить Ctrl
                inputs[3].ki.wVk       = VK_CONTROL
                inputs[3].ki.dwFlags   = KEYEVENTF_KEYUP

                ctypes.windll.user32.SendInput(4, inputs, ctypes.sizeof(_INPUT))
            except Exception:  # noqa: BLE001
                pass  # вставка не критична — текст уже в буфере обмена

        # ── Формируем результат ───────────────────────────────────────────────
        result = PipelineResult(
            text               = final_text,
            language           = stt_result.language,
            audio_duration_sec = stt_result.audio_duration_sec,
            transcribe_sec     = stt_result.transcribe_time_sec,
            edit_sec           = edit_result.edit_time_sec,
            total_sec          = round(total_sec, 3),
            llm_used           = edit_result.success,
            segments           = stt_result.segments,
        )

        self._set_state(PipelineState.DONE)

        if self.on_result:
            self.on_result(final_text, result)

        # Возвращаемся в IDLE через 2 сек (иконка успеха держится)
        threading.Timer(2.0, self._return_to_idle).start()

    # ── Вспомогательные ───────────────────────────────────────────────────────

    def _finish_error(self, message: str) -> None:
        self._last_error = message
        self._set_state(PipelineState.ERROR)
        if self.on_error:
            self.on_error(message)
        threading.Timer(5.0, self._return_to_idle).start()

    def _return_to_idle(self) -> None:
        with self._state_lock:
            if self._state in (PipelineState.DONE, PipelineState.ERROR):
                self._set_state_locked(PipelineState.IDLE)

    def _set_state(self, state: PipelineState) -> None:
        with self._state_lock:
            self._set_state_locked(state)

    def _set_state_locked(self, state: PipelineState) -> None:
        """Устанавливает состояние. Вызывать только под _state_lock."""
        self._state = state
        if self.on_state_change:
            # Callback вызываем вне лока чтобы избежать deadlock
            threading.Thread(
                target=self.on_state_change, args=(state,), daemon=True
            ).start()


# ── Консольный тест: python -m core.pipeline ─────────────────────────────────

def _load_config() -> dict:
    import yaml  # type: ignore
    p = Path(__file__).parent.parent / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"config.yaml не найден: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Консольный MVP:
      1. Загружаем все модели с прогрессом
      2. Цикл: Enter → запись → Enter → обработка → результат
      3. Ctrl+C — выход
    """
    import sys

    config = _load_config()
    mode   = config.get("hotkey", {}).get("mode", "hold")

    print("\n" + "═" * 64)
    print("  VoiceScribe — консольный тест пайплайна")
    print("═" * 64)
    print(f"\n  Режим: {mode.upper()}")
    print(f"  STT:   {config['transcriber']['model']} / "
          f"{config['transcriber']['device']} / "
          f"{config['transcriber']['compute_type']}")
    print(f"  LLM:   {'включён' if config['editor'].get('enabled', True) else 'отключён'}\n")

    pipeline = Pipeline(config)

    # ── Прогресс загрузки ─────────────────────────────────────────────────────
    print("Загружаем модели:")

    def _on_progress(msg: str, frac: float) -> None:
        bar_w = 30
        filled = int(frac * bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)
        sys.stdout.write(f"\r  [{bar}] {msg:<35}")
        sys.stdout.flush()
        if frac >= 1.0:
            print()

    t0 = time.perf_counter()
    try:
        pipeline.load_models(on_progress=_on_progress)
    except Exception as exc:
        print(f"\n  [ERR] Не удалось загрузить модели: {exc}")
        return
    print(f"  Готово за {time.perf_counter() - t0:.1f} с\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    _icons = {
        PipelineState.IDLE:       "○",
        PipelineState.RECORDING:  "◉",
        PipelineState.PROCESSING: "◌",
        PipelineState.DONE:       "✓",
        PipelineState.ERROR:      "✗",
    }

    def _on_state(state: PipelineState) -> None:
        icon = _icons.get(state, "?")
        sys.stdout.write(f"\r  {icon} {state.name:<12}          \n")
        sys.stdout.flush()

    def _on_result(text: str, result: PipelineResult) -> None:
        print("\n" + "─" * 64)
        print(f"  {text}")
        print("─" * 64)
        print(f"  {result.summary()}")
        if result.segments:
            for s in result.segments:
                print(f"    [{s['start']:5.2f}–{s['end']:5.2f}] {s['text']}")
        print(f"\n  ✓ Скопировано в буфер обмена.\n")

    def _on_error(msg: str) -> None:
        print(f"\n  ✗ {msg}\n")

    def _on_tick(elapsed: float) -> None:
        sys.stdout.write(f"\r  ◉ RECORDING  {elapsed:.1f} с          ")
        sys.stdout.flush()

    pipeline.on_state_change   = _on_state
    pipeline.on_result         = _on_result
    pipeline.on_error          = _on_error
    pipeline.on_recording_tick = _on_tick

    # ── Основной цикл ─────────────────────────────────────────────────────────
    print("─" * 64)
    if mode == "hold":
        print("  Hold-режим: ENTER — начать, ENTER — остановить.")
    else:
        print("  Toggle-режим: ENTER — начать/остановить.")
        print(f"  Авто-стоп по тишине: {config['hotkey'].get('autostop_sec', 30)} с")
    print("  Ctrl+C — выход.")
    print("─" * 64 + "\n")

    session = 0
    try:
        while True:
            input(f"  [{session + 1}] Нажмите ENTER для начала записи ... ")
            started = pipeline.start_recording()
            if not started:
                print("  Пайплайн занят. Подождите.")
                continue
            session += 1

            if mode == "hold":
                try:
                    input()  # ждём Enter для остановки
                except EOFError:
                    pass
                pipeline.stop_recording()
            else:
                # toggle: ждём второго Enter или авто-стоп
                try:
                    input("  Нажмите ENTER для остановки ...\n")
                except EOFError:
                    pass
                pipeline.stop_recording()

            # Ждём завершения обработки
            if pipeline._process_thread:
                pipeline._process_thread.join(timeout=60)

    except KeyboardInterrupt:
        print("\n\n  Выход.")
        if pipeline.state == PipelineState.RECORDING:
            pipeline.stop_recording()


if __name__ == "__main__":
    main()
