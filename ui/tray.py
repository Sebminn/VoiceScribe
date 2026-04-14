"""
ui/tray.py — системный трей, иконки, меню, горячие клавиши.

Состояния иконки:
    🔵 IDLE       — ожидание (синяя)
    🔴 RECORDING  — запись (красная, пульсирует)
    🟡 PROCESSING — обработка (жёлтая)
    🟢 DONE       — готово, скопировано (зелёная, 2 сек)
    ⚪ MUTED      — пауза / горячая клавиша отключена (серая)
    🔴❗ ERROR    — ошибка (красная + бейдж "!")

Контекстное меню (ПКМ на иконке):
    Режим: Удержание (Hold)  ✓
    Режим: Свободные руки (Toggle)
    ──
    Горячая клавиша...
    Настройки...
    ──
    Пауза / Возобновить
    ──
    Выход
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QAction, QActionGroup, QBrush, QColor, QFont, QIcon, QPainter, QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QLabel, QLineEdit,
    QMenu, QPushButton, QSystemTrayIcon, QVBoxLayout,
)


# ── Генерация иконок ──────────────────────────────────────────────────────────

def _make_pixmap(
    color_hex: str,
    size: int = 64,
    dim: bool = False,
    badge: str = "",
) -> QPixmap:
    """Рисует круглую цветную иконку, опционально с бейджем '!'."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    color = QColor(color_hex)
    if dim:
        color.setAlpha(120)

    m = max(2, size // 10)
    painter.setBrush(QBrush(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(m, m, size - 2 * m, size - 2 * m)

    if badge:
        bs = size // 3
        bx, by = size - bs - 1, 1
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawEllipse(bx, by, bs, bs)
        painter.setPen(QColor(color_hex))
        fnt = QFont("Arial", max(8, bs // 2 + 1))
        fnt.setBold(True)
        painter.setFont(fnt)
        painter.drawText(bx, by, bs, bs, Qt.AlignmentFlag.AlignCenter, badge)

    painter.end()
    return pixmap


def _build_icons(resources_dir: Path) -> dict[str, QIcon]:
    """
    Строит словарь QIcon для всех состояний.
    Если в resources/ есть готовые PNG — использует их; иначе генерирует.
    """
    specs = {
        "idle":            ("#2563EB", False, ""),
        "recording":       ("#EF4444", False, ""),
        "recording_dim":   ("#EF4444", True,  ""),
        "processing":      ("#F59E0B", False, ""),
        "done":            ("#22C55E", False, ""),
        "muted":           ("#94A3B8", False, ""),
        "error":           ("#EF4444", False, "!"),
    }
    icons: dict[str, QIcon] = {}
    resources_dir.mkdir(exist_ok=True)

    for name, (color, dim, badge) in specs.items():
        png_path = resources_dir / f"icon_{name}.png"
        if png_path.exists():
            icons[name] = QIcon(str(png_path))
        else:
            px = _make_pixmap(color, size=64, dim=dim, badge=badge)
            px.save(str(png_path), "PNG")
            icons[name] = QIcon(px)

    return icons


# ── Qt-сигнальный мост (thread-safe колбэки → главный поток) ─────────────────

class _Signals(QObject):
    """Мост между background-потоками Pipeline и главным Qt-потоком."""
    state_changed  = pyqtSignal(object)       # PipelineState
    result_ready   = pyqtSignal(str, object)  # text, PipelineResult
    error          = pyqtSignal(str)
    tick           = pyqtSignal(float)
    progress       = pyqtSignal(str, float)   # msg, fraction 0–1
    models_ready   = pyqtSignal()
    models_error   = pyqtSignal(str)


# ── Загрузка моделей в QThread ────────────────────────────────────────────────

class _ModelLoader(QThread):
    def __init__(self, pipeline, signals: _Signals) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._signals  = signals

    def run(self) -> None:
        try:
            self._pipeline.load_models(
                on_progress=lambda m, f: self._signals.progress.emit(m, f)
            )
            self._signals.models_ready.emit()
        except Exception as exc:  # noqa: BLE001
            self._signals.models_error.emit(str(exc))


# ── Горячая клавиша — парсер и листенер ──────────────────────────────────────

def _normalize_key(key) -> str:
    """pynput key → строка ('ctrl', 'shift', 'space', 'a', 'f1', …)."""
    try:
        from pynput.keyboard import Key  # type: ignore
        _MOD_MAP = {
            Key.ctrl_l: "ctrl",  Key.ctrl_r: "ctrl",
            Key.shift_l: "shift", Key.shift_r: "shift",
            Key.alt_l: "alt",    Key.alt_r: "alt",
            Key.alt_gr: "alt",
            Key.cmd:   "win",    Key.cmd_l: "win",   Key.cmd_r: "win",
        }
        if key in _MOD_MAP:
            return _MOD_MAP[key]
        if isinstance(key, Key):
            return key.name  # 'space', 'enter', 'f1', …
    except Exception:  # noqa: BLE001
        pass
    # KeyCode
    if hasattr(key, "char") and key.char:
        return key.char.lower()
    return str(key)


def _parse_hotkey(hotkey_str: str) -> tuple[frozenset[str], str]:
    """
    'ctrl+shift+space' → (frozenset({'ctrl', 'shift'}), 'space')
    """
    _MODIFIER_NAMES = {"ctrl", "shift", "alt", "win"}
    parts = [p.strip().lower() for p in hotkey_str.split("+")]
    mods  = frozenset(p for p in parts if p in _MODIFIER_NAMES)
    mains = [p for p in parts if p not in _MODIFIER_NAMES]
    return mods, (mains[0] if mains else "")


class HotkeyListener:
    """
    Глобальный перехватчик горячей клавиши через pynput.
    Поддерживает hold-режим (on_activate + on_deactivate)
    и toggle-режим (только on_activate).
    """

    def __init__(
        self,
        hotkey_str: str,
        on_activate:   Callable[[], None],
        on_deactivate: Optional[Callable[[], None]] = None,
    ) -> None:
        self._on_activate   = on_activate
        self._on_deactivate = on_deactivate
        self._paused        = False
        self._hotkey_down   = False
        self._active_mods:  set[str] = set()
        self._required_mods: frozenset[str]
        self._main_key:      str
        self._required_mods, self._main_key = _parse_hotkey(hotkey_str)
        self._listener = None

    # ── Управление ────────────────────────────────────────────────────────────

    def start(self) -> None:
        try:
            from pynput.keyboard import Listener  # type: ignore
            self._listener = Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] pynput listener не запущен: {exc}", file=sys.stderr)

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:  # noqa: BLE001
                pass
            self._listener = None
        self._active_mods.clear()
        self._hotkey_down = False

    def update_hotkey(self, hotkey_str: str) -> None:
        """Меняет горячую клавишу без перезапуска (безопасно во время работы)."""
        self._required_mods, self._main_key = _parse_hotkey(hotkey_str)
        self._hotkey_down = False

    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._paused = value
        if value:
            self._hotkey_down = False

    # ── Обработчики pynput ────────────────────────────────────────────────────

    def _on_press(self, key) -> None:
        norm = _normalize_key(key)
        if norm in ("ctrl", "shift", "alt", "win"):
            self._active_mods.add(norm)
            return
        if (
            norm == self._main_key
            and self._required_mods <= self._active_mods
            and not self._hotkey_down
            and not self._paused
        ):
            self._hotkey_down = True
            try:
                self._on_activate()
            except Exception:  # noqa: BLE001
                pass

    def _on_release(self, key) -> None:
        norm = _normalize_key(key)
        if norm in ("ctrl", "shift", "alt", "win"):
            self._active_mods.discard(norm)
        elif norm == self._main_key and self._hotkey_down:
            self._hotkey_down = False
            if self._on_deactivate and not self._paused:
                try:
                    self._on_deactivate()
                except Exception:  # noqa: BLE001
                    pass


# ── Диалог переназначения горячей клавиши ─────────────────────────────────────

def _qt_key_name(key: Qt.Key) -> str:
    """Qt.Key → строка в формате конфига ('space', 'f1', 'a', …)."""
    _SPECIAL = {
        Qt.Key.Key_Space:     "space",
        Qt.Key.Key_Tab:       "tab",
        Qt.Key.Key_Return:    "return",
        Qt.Key.Key_Enter:     "return",
        Qt.Key.Key_Escape:    "esc",
        Qt.Key.Key_Backspace: "backspace",
        Qt.Key.Key_Delete:    "delete",
        Qt.Key.Key_Insert:    "insert",
        Qt.Key.Key_Home:      "home",
        Qt.Key.Key_End:       "end",
        Qt.Key.Key_PageUp:    "page_up",
        Qt.Key.Key_PageDown:  "page_down",
        Qt.Key.Key_Up:        "up",
        Qt.Key.Key_Down:      "down",
        Qt.Key.Key_Left:      "left",
        Qt.Key.Key_Right:     "right",
    }
    for i in range(1, 13):
        _SPECIAL[getattr(Qt.Key, f"Key_F{i}")] = f"f{i}"

    if key in _SPECIAL:
        return _SPECIAL[key]
    seq = Qt.Key(key)
    text = seq.name if hasattr(seq, "name") else ""
    if text.startswith("Key_"):
        text = text[4:]
    return text.lower() if text else ""


class HotkeyDialog(QDialog):
    """Мини-диалог захвата новой горячей клавиши."""

    def __init__(self, current_hotkey: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Горячая клавиша — VoiceScribe")
        self.setFixedSize(340, 170)
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowType.WindowContextHelpButtonHint
        )

        self._captured   = current_hotkey
        self._capturing  = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        layout.addWidget(QLabel(f"Текущая комбинация: <b>{current_hotkey}</b>"))

        self._capture_btn = QPushButton("▶  Нажмите для записи новой комбинации")
        self._capture_btn.setCheckable(True)
        self._capture_btn.toggled.connect(self._toggle_capture)
        layout.addWidget(self._capture_btn)

        self._preview = QLabel("Новая: —")
        self._preview.setStyleSheet("color: #555; font-size: 13px;")
        layout.addWidget(self._preview)

        self._info = QLabel("")
        self._info.setStyleSheet("color: #999; font-size: 11px;")
        layout.addWidget(self._info)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def result_hotkey(self) -> str:
        return self._captured

    def _toggle_capture(self, checked: bool) -> None:
        self._capturing = checked
        if checked:
            self._capture_btn.setText("⏺  Нажмите комбинацию клавиш ...")
            self._info.setText("")
            self.grabKeyboard()
        else:
            self._capture_btn.setText("▶  Нажмите для записи новой комбинации")
            self.releaseKeyboard()

    def keyPressEvent(self, event) -> None:
        if not self._capturing:
            super().keyPressEvent(event)
            return

        key = Qt.Key(event.key())
        _PURE_MODIFIERS = {
            Qt.Key.Key_Control, Qt.Key.Key_Shift,
            Qt.Key.Key_Alt,     Qt.Key.Key_Meta,
        }
        if key in _PURE_MODIFIERS:
            return  # ждём основную клавишу

        modifiers: list[str] = []
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            modifiers.append("ctrl")
        if mods & Qt.KeyboardModifier.ShiftModifier:
            modifiers.append("shift")
        if mods & Qt.KeyboardModifier.AltModifier:
            modifiers.append("alt")
        if mods & Qt.KeyboardModifier.MetaModifier:
            modifiers.append("win")

        key_name = _qt_key_name(key)
        if not key_name:
            self._info.setText("Клавиша не распознана, попробуйте другую.")
            return

        # Запрещаем системные комбинации
        combo = "+".join(modifiers + [key_name])
        _FORBIDDEN = {"ctrl+c", "ctrl+v", "ctrl+z", "ctrl+x",
                      "alt+f4", "ctrl+alt+delete"}
        if combo.lower() in _FORBIDDEN:
            self._info.setText(f"Комбинация '{combo}' зарезервирована системой.")
            return

        self._captured = combo
        self._preview.setText(f"Новая: <b>{combo}</b>")
        self._capturing = False
        self._capture_btn.setChecked(False)
        self._capture_btn.setText("▶  Нажмите для записи новой комбинации")
        self.releaseKeyboard()
        event.accept()


# ── Главный класс трея ────────────────────────────────────────────────────────

class VoiceScribeTray(QObject):
    """
    Системный трей VoiceScribe.

    Управляет Pipeline, HotkeyListener, иконками и меню.
    Все обновления UI происходят в главном Qt-потоке через сигналы.
    """

    def __init__(self, app: QApplication, config: dict,
                 config_manager=None, pipeline=None) -> None:
        super().__init__()
        self._app    = app
        self._config = config
        self._cm     = config_manager   # core.config.ConfigManager или None
        self._paused = False
        self._loading_done = False
        self._recording_sec = 0.0

        self._signals = _Signals()

        # Pipeline: принимаем уже загруженный (из onboarding) или создаём новый
        from core.pipeline import Pipeline, PipelineState  # noqa: PLC0415
        self._PipelineState = PipelineState
        self._models_preloaded = pipeline is not None
        self._pipeline = pipeline if pipeline is not None else Pipeline(config)
        self._pipeline.on_state_change   = self._signals.state_changed.emit
        self._pipeline.on_result         = self._signals.result_ready.emit
        self._pipeline.on_error          = self._signals.error.emit
        self._pipeline.on_recording_tick = self._signals.tick.emit

        # Иконки и анимация
        self._icons: dict[str, QIcon] = {}
        self._pulse_timer  = QTimer(self)
        self._pulse_state  = False

        # Трей и меню
        self._tray:         QSystemTrayIcon
        self._action_hold:  QAction
        self._action_toggle: QAction
        self._action_pause: QAction

        # Горячая клавиша
        self._hotkey_listener: Optional[HotkeyListener] = None

    # ── Инициализация ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Создаёт трей, запускает загрузку моделей и слушатель клавиш."""
        resources_dir = Path(__file__).parent.parent / "resources"
        self._icons = _build_icons(resources_dir)

        self._create_tray()
        self._connect_signals()
        self._setup_hotkey()
        self._start_model_loading()

    def _create_tray(self) -> None:
        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(self._icons["idle"])
        self._tray.setToolTip("VoiceScribe — загрузка моделей...")
        self._tray.setVisible(True)
        self._tray.activated.connect(self._on_tray_activated)

        menu = QMenu()

        # ── Режим ─────────────────────────────────────────────────────────────
        mode_group = QActionGroup(menu)
        mode_group.setExclusive(True)

        self._action_hold = QAction("Режим: Удержание (Hold)", menu)
        self._action_hold.setCheckable(True)
        self._action_hold.triggered.connect(lambda: self._set_mode("hold"))
        mode_group.addAction(self._action_hold)
        menu.addAction(self._action_hold)

        self._action_toggle = QAction("Режим: Свободные руки (Toggle)", menu)
        self._action_toggle.setCheckable(True)
        self._action_toggle.triggered.connect(lambda: self._set_mode("toggle"))
        mode_group.addAction(self._action_toggle)
        menu.addAction(self._action_toggle)

        current_mode = self._config.get("hotkey", {}).get("mode", "hold")
        (self._action_hold if current_mode == "hold" else self._action_toggle).setChecked(True)

        menu.addSeparator()

        # ── Прочие пункты ─────────────────────────────────────────────────────
        menu.addAction("Горячая клавиша...", self._open_hotkey_dialog)
        menu.addAction("Настройки...",       self._open_settings)

        menu.addSeparator()

        self._action_pause = QAction("Пауза", menu)
        self._action_pause.triggered.connect(self._toggle_pause)
        menu.addAction(self._action_pause)

        menu.addSeparator()
        menu.addAction("Выход", self._quit)

        self._tray.setContextMenu(menu)

        # Пульс-таймер (для анимации записи)
        self._pulse_timer.setInterval(600)
        self._pulse_timer.timeout.connect(self._pulse_tick)

    def _connect_signals(self) -> None:
        self._signals.state_changed.connect(self._on_state_changed)
        self._signals.result_ready .connect(self._on_result_ready)
        self._signals.error        .connect(self._on_error)
        self._signals.tick         .connect(self._on_tick)
        self._signals.progress     .connect(self._on_progress)
        self._signals.models_ready .connect(self._on_models_ready)
        self._signals.models_error .connect(self._on_models_error)

    def _setup_hotkey(self) -> None:
        hotkey_str = self._config.get("hotkey", {}).get("key", "ctrl+shift+space")
        mode       = self._config.get("hotkey", {}).get("mode", "hold")

        if mode == "hold":
            self._hotkey_listener = HotkeyListener(
                hotkey_str,
                on_activate   = self._pipeline.start_recording,
                on_deactivate = self._pipeline.stop_recording,
            )
        else:
            self._hotkey_listener = HotkeyListener(
                hotkey_str,
                on_activate = self._pipeline.toggle_recording,
            )

        # Слушатель отключён пока модели не загружены
        self._hotkey_listener.paused = True
        self._hotkey_listener.start()

    def _start_model_loading(self) -> None:
        if self._models_preloaded:
            # Модели уже загружены в onboarding — сразу сигналим о готовности
            QTimer.singleShot(0, self._on_models_ready)
            return
        self._loader = _ModelLoader(self._pipeline, self._signals)
        self._loader.start()

    # ── Слоты сигналов ────────────────────────────────────────────────────────

    @pyqtSlot(str, float)
    def _on_progress(self, message: str, fraction: float) -> None:
        self._tray.setToolTip(f"VoiceScribe — {message}")

    @pyqtSlot()
    def _on_models_ready(self) -> None:
        self._loading_done = True
        if self._hotkey_listener and not self._paused:
            self._hotkey_listener.paused = False
        self._update_tooltip()
        self._tray.setIcon(self._icons["idle"])

    @pyqtSlot(str)
    def _on_models_error(self, message: str) -> None:
        self._tray.setIcon(self._icons["error"])
        self._tray.setToolTip(f"VoiceScribe — ошибка загрузки: {message}")
        self._tray.showMessage(
            "VoiceScribe — ошибка",
            f"Не удалось загрузить модели:\n{message}",
            QSystemTrayIcon.MessageIcon.Critical,
            5000,
        )

    @pyqtSlot(object)
    def _on_state_changed(self, state) -> None:
        self._update_icon(state)

    @pyqtSlot(str, object)
    def _on_result_ready(self, text: str, result) -> None:
        preview = text[:self._config.get("output", {}).get("notify_preview_chars", 60)]
        if self._config.get("output", {}).get("notify", True):
            self._tray.showMessage(
                "VoiceScribe — скопировано!",
                preview,
                QSystemTrayIcon.MessageIcon.Information,
                self._config.get("output", {}).get("notify_duration_ms", 3000),
            )

    @pyqtSlot(str)
    def _on_error(self, message: str) -> None:
        self._tray.showMessage(
            "VoiceScribe",
            message,
            QSystemTrayIcon.MessageIcon.Warning,
            4000,
        )

    @pyqtSlot(float)
    def _on_tick(self, elapsed: float) -> None:
        self._recording_sec = elapsed
        self._tray.setToolTip(f"VoiceScribe — запись {elapsed:.0f} с")

    # ── Иконка и анимация ─────────────────────────────────────────────────────

    def _update_icon(self, state) -> None:
        PS = self._PipelineState
        icon_map = {
            PS.IDLE:       "idle",
            PS.PROCESSING: "processing",
            PS.DONE:       "done",
            PS.ERROR:      "error",
        }
        if state == PS.RECORDING:
            self._start_pulse()
        else:
            self._stop_pulse()
            key = icon_map.get(state, "idle")
            icon = self._icons["muted"] if self._paused else self._icons[key]
            self._tray.setIcon(icon)

        # Тултип
        if state == PS.PROCESSING:
            self._tray.setToolTip("VoiceScribe — распознаю...")
        elif state == PS.DONE:
            self._tray.setToolTip("VoiceScribe — скопировано!")
        elif state == PS.ERROR:
            self._tray.setToolTip(f"VoiceScribe — ошибка: {self._pipeline.last_error}")
        elif state == PS.IDLE:
            self._update_tooltip()

    def _start_pulse(self) -> None:
        self._tray.setIcon(self._icons["recording"])
        self._pulse_state = False
        self._pulse_timer.start()

    def _stop_pulse(self) -> None:
        self._pulse_timer.stop()

    def _pulse_tick(self) -> None:
        self._pulse_state = not self._pulse_state
        key = "recording_dim" if self._pulse_state else "recording"
        self._tray.setIcon(self._icons[key])

    def _update_tooltip(self) -> None:
        if not self._loading_done:
            self._tray.setToolTip("VoiceScribe — загрузка моделей...")
            return
        mode  = self._config.get("hotkey", {}).get("mode", "hold")
        key   = self._config.get("hotkey", {}).get("key", "ctrl+shift+space")
        label = "Hold" if mode == "hold" else "Toggle"
        if self._paused:
            self._tray.setToolTip(f"VoiceScribe — ПАУЗА ({key})")
        else:
            self._tray.setToolTip(f"VoiceScribe — готов [{label}] {key}")

    # ── Действия меню ─────────────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        if mode == self._config["hotkey"]["mode"]:
            return
        self._config["hotkey"]["mode"] = mode
        self._pipeline.mode = mode
        (self._action_hold if mode == "hold" else self._action_toggle).setChecked(True)

        # Перезапускаем listener с новым поведением
        if self._hotkey_listener:
            self._hotkey_listener.stop()

        hotkey_str = self._config["hotkey"]["key"]
        if mode == "hold":
            self._hotkey_listener = HotkeyListener(
                hotkey_str,
                on_activate   = self._pipeline.start_recording,
                on_deactivate = self._pipeline.stop_recording,
            )
        else:
            self._hotkey_listener = HotkeyListener(
                hotkey_str,
                on_activate = self._pipeline.toggle_recording,
            )
        self._hotkey_listener.paused = self._paused or not self._loading_done
        self._hotkey_listener.start()

        self._save_config()
        self._update_tooltip()

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._hotkey_listener:
            self._hotkey_listener.paused = self._paused or not self._loading_done
        self._action_pause.setText("Возобновить" if self._paused else "Пауза")
        self._tray.setIcon(self._icons["muted"] if self._paused else self._icons["idle"])
        self._update_tooltip()

    def _open_hotkey_dialog(self) -> None:
        current = self._config.get("hotkey", {}).get("key", "ctrl+shift+space")
        dialog  = HotkeyDialog(current)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        new_hotkey = dialog.result_hotkey()
        if new_hotkey == current or not new_hotkey:
            return

        self._config["hotkey"]["key"] = new_hotkey
        if self._hotkey_listener:
            self._hotkey_listener.update_hotkey(new_hotkey)
        self._save_config()
        self._update_tooltip()

    def _open_settings(self) -> None:
        from ui.settings import SettingsDialog  # lazy — нет циклической зависимости
        dlg = SettingsDialog(
            config=self._config,
            config_manager=self._cm,
            on_applied=self._on_settings_applied,
        )
        dlg.exec()

    def _on_settings_applied(self, changed_keys: set) -> None:
        """Горячее применение изменённых настроек без перезапуска."""
        hotkey_keys = {"hotkey.key", "hotkey.mode",
                       "hotkey.min_hold_ms", "hotkey.autostop_sec"}
        if changed_keys & hotkey_keys:
            # Перезапустить слушатель с обновлённым режимом / клавишей
            if self._hotkey_listener:
                self._hotkey_listener.stop()
            self._setup_hotkey()

        # Синхронизируем галочки режима в меню
        mode = self._config.get("hotkey", {}).get("mode", "hold")
        self._action_hold.setChecked(mode == "hold")
        self._action_toggle.setChecked(mode == "toggle")
        self._pipeline.mode = mode

        self._update_tooltip()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Двойной клик → если есть ошибка, показать её."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            err = self._pipeline.last_error
            if err:
                self._tray.showMessage(
                    "VoiceScribe — последняя ошибка",
                    err,
                    QSystemTrayIcon.MessageIcon.Warning,
                    4000,
                )

    def _save_config(self) -> None:
        """Сохраняет изменения в config.yaml."""
        if self._cm is not None:
            try:
                self._cm.save()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] ConfigManager.save: {exc}", file=sys.stderr)
            return
        # Fallback: inline ruamel.yaml (если ConfigManager не передан)
        try:
            from ruamel.yaml import YAML  # type: ignore
            config_path = Path(__file__).parent.parent / "config.yaml"
            yaml = YAML()
            yaml.preserve_quotes = True
            with open(config_path, encoding="utf-8") as f:
                data = yaml.load(f)
            data["hotkey"]["key"]  = self._config["hotkey"]["key"]
            data["hotkey"]["mode"] = self._config["hotkey"]["mode"]
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Не удалось сохранить config.yaml: {exc}", file=sys.stderr)

    def _quit(self) -> None:
        if self._hotkey_listener:
            self._hotkey_listener.stop()
        self._tray.setVisible(False)
        self._app.quit()
