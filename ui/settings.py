"""
ui/settings.py — окно настроек VoiceScribe.

Открывается из меню трея → «Настройки...»
Четыре вкладки согласно документации (раздел 6):
    1. Запись   — режим, горячая клавиша, таймауты
    2. Аудио    — микрофон, параметры потока (readonly)
    3. Модели   — Whisper + Phi-3.5 Mini
    4. Вывод    — уведомления, длина превью

Принцип работы:
    • OK / Apply  — валидировать → обновить config (in-place) → сохранить
                    → вызвать on_applied(changed_keys) для горячего применения
    • Cancel      — закрыть без сохранения
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSizePolicy, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _form_group(title: str) -> tuple[QGroupBox, QFormLayout]:
    """Создаёт QGroupBox с QFormLayout внутри."""
    box = QGroupBox(title)
    form = QFormLayout(box)
    form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
    form.setContentsMargins(12, 8, 12, 12)
    form.setSpacing(8)
    return box, form


def _readonly(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #888;")
    return lbl


def _hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #aaa; font-size: 11px;")
    return lbl


# ── Диалог настроек ───────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """
    Окно настроек.

    Параметры:
        config         — живой словарь (CommentedMap из ConfigManager.data)
        config_manager — ConfigManager для сохранения (save())
        on_applied     — callback(changed_keys: set[str]) для горячего применения
    """

    # Ключи, при изменении которых требуется перезапуск приложения
    _RESTART_KEYS = frozenset({
        "transcriber.model", "transcriber.device", "transcriber.compute_type",
        "editor.model_path", "editor.gpu_layers", "editor.context_size",
    })

    def __init__(
        self,
        config,
        config_manager=None,
        on_applied: Optional[Callable[[set], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config  = config
        self._cm      = config_manager
        self._on_applied = on_applied

        self.setWindowTitle("Настройки — VoiceScribe")
        self.setMinimumWidth(500)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )

        # Виджеты (заполняются в _build_*)
        self._w_mode:           QComboBox
        self._w_hotkey:         QLineEdit
        self._w_min_hold:       QSpinBox
        self._w_autostop:       QSpinBox
        self._w_max_record:     QSpinBox
        self._w_device:         QComboBox
        self._w_stt_model:      QComboBox
        self._w_stt_device:     QComboBox
        self._w_stt_compute:    QComboBox
        self._w_beam_size:      QSpinBox
        self._w_language:       QComboBox
        self._w_editor_enabled: QCheckBox
        self._w_model_path:     QLineEdit
        self._w_gpu_layers:     QSpinBox
        self._w_temperature:    QDoubleSpinBox
        self._w_timeout:        QSpinBox
        self._w_notify:         QCheckBox
        self._w_notify_ms:      QSpinBox
        self._w_preview_chars:  QSpinBox

        self._build_ui()
        self._load_values()

    # ── Построение UI ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(self._build_recording_tab(), "Запись")
        tabs.addTab(self._build_audio_tab(),     "Аудио")
        tabs.addTab(self._build_models_tab(),    "Модели")
        tabs.addTab(self._build_output_tab(),    "Уведомления")
        root.addWidget(tabs)

        # Кнопки OK / Apply / Cancel
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self._on_apply
        )
        root.addWidget(btns)

    # ── Вкладка «Запись» ──────────────────────────────────────────────────────

    def _build_recording_tab(self) -> QWidget:
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(4, 4, 4, 4)

        # Горячая клавиша и режим
        box1, f1 = _form_group("Горячая клавиша")
        self._w_hotkey = QLineEdit()
        self._w_hotkey.setReadOnly(True)
        self._w_hotkey.setPlaceholderText("ctrl+shift+space")
        btn_change = QPushButton("Изменить…")
        btn_change.setFixedWidth(90)
        btn_change.clicked.connect(self._open_hotkey_dialog)
        hk_row = QHBoxLayout()
        hk_row.addWidget(self._w_hotkey)
        hk_row.addWidget(btn_change)
        f1.addRow("Комбинация:", hk_row)

        self._w_mode = QComboBox()
        self._w_mode.addItem("Удержание (Hold)",        "hold")
        self._w_mode.addItem("Свободные руки (Toggle)", "toggle")
        f1.addRow("Режим записи:", self._w_mode)
        vbox.addWidget(box1)

        # Hold-параметры
        box2, f2 = _form_group("Hold — параметры удержания")
        self._w_min_hold = QSpinBox()
        self._w_min_hold.setRange(100, 5000)
        self._w_min_hold.setSuffix(" мс")
        self._w_min_hold.setSingleStep(100)
        f2.addRow("Мин. время нажатия:", self._w_min_hold)
        f2.addRow("", _hint("Нажатия короче этого времени игнорируются"))
        vbox.addWidget(box2)

        # Toggle-параметры
        box3, f3 = _form_group("Toggle — авто-стоп")
        self._w_autostop = QSpinBox()
        self._w_autostop.setRange(5, 300)
        self._w_autostop.setSuffix(" с")
        f3.addRow("Тишина для авто-стопа:", self._w_autostop)

        self._w_max_record = QSpinBox()
        self._w_max_record.setRange(10, 600)
        self._w_max_record.setSuffix(" с")
        f3.addRow("Макс. длина записи:", self._w_max_record)
        vbox.addWidget(box3)

        vbox.addStretch()
        return page

    # ── Вкладка «Аудио» ───────────────────────────────────────────────────────

    def _build_audio_tab(self) -> QWidget:
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(4, 4, 4, 4)

        box, form = _form_group("Устройство ввода")

        self._w_device = QComboBox()
        self._w_device.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._populate_devices()
        form.addRow("Микрофон:", self._w_device)

        form.addRow("Частота:", _readonly("16 000 Гц — не менять (требование Whisper)"))
        form.addRow("Каналы:", _readonly("Моно — не менять"))
        vbox.addWidget(box)
        vbox.addStretch()
        return page

    def _populate_devices(self) -> None:
        self._w_device.addItem("Системный по умолчанию", "default")
        try:
            from core.audio import AudioCapture  # noqa: PLC0415
            for d in AudioCapture.list_devices():
                self._w_device.addItem(f"[{d['index']}] {d['name']}", str(d["index"]))
        except Exception:  # noqa: BLE001
            pass

    # ── Вкладка «Модели» ──────────────────────────────────────────────────────

    def _build_models_tab(self) -> QWidget:
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(4, 4, 4, 4)

        # Whisper
        box1, f1 = _form_group("Распознавание речи (Faster-Whisper)")

        self._w_stt_model = QComboBox()
        for m in ["tiny", "base", "small", "medium",
                  "large-v2", "large-v3", "large-v3-turbo"]:
            self._w_stt_model.addItem(m, m)
        f1.addRow("Модель:", self._w_stt_model)

        self._w_stt_device = QComboBox()
        for d in [("CUDA (GPU)", "cuda"), ("CPU", "cpu")]:
            self._w_stt_device.addItem(d[0], d[1])
        f1.addRow("Устройство:", self._w_stt_device)

        self._w_stt_compute = QComboBox()
        for ct in [("int8 — быстро (~1.5 ГБ)", "int8"),
                   ("float16 — точнее (~3 ГБ)", "float16"),
                   ("int8_float16", "int8_float16")]:
            self._w_stt_compute.addItem(ct[0], ct[1])
        f1.addRow("Тип вычислений:", self._w_stt_compute)

        self._w_beam_size = QSpinBox()
        self._w_beam_size.setRange(1, 10)
        f1.addRow("Ширина поиска (beam):", self._w_beam_size)

        self._w_language = QComboBox()
        for lang in [("Авто (RU/EN)", ""), ("Русский", "ru"), ("English", "en"),
                     ("Deutsch", "de"), ("Français", "fr"), ("Español", "es")]:
            self._w_language.addItem(lang[0], lang[1])
        f1.addRow("Язык:", self._w_language)
        f1.addRow("", _hint("Изменения модели применяются после перезапуска"))
        vbox.addWidget(box1)

        # Phi-3.5 Mini
        box2, f2 = _form_group("Постобработка (Phi-3.5 Mini)")

        self._w_editor_enabled = QCheckBox("Включить LLM постобработку")
        f2.addRow("", self._w_editor_enabled)

        path_row = QHBoxLayout()
        self._w_model_path = QLineEdit()
        self._w_model_path.setPlaceholderText("models/Phi-3.5-mini-Q4.gguf")
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse_model_path)
        path_row.addWidget(self._w_model_path)
        path_row.addWidget(btn_browse)
        f2.addRow("Путь к модели:", path_row)

        self._w_gpu_layers = QSpinBox()
        self._w_gpu_layers.setRange(-1, 128)
        self._w_gpu_layers.setSpecialValueText("-1  (все на GPU)")
        f2.addRow("GPU слои:", self._w_gpu_layers)

        self._w_temperature = QDoubleSpinBox()
        self._w_temperature.setRange(0.0, 1.0)
        self._w_temperature.setSingleStep(0.05)
        self._w_temperature.setDecimals(2)
        f2.addRow("Температура:", self._w_temperature)

        self._w_timeout = QSpinBox()
        self._w_timeout.setRange(1, 60)
        self._w_timeout.setSuffix(" с")
        f2.addRow("Таймаут:", self._w_timeout)
        f2.addRow("", _hint("При превышении таймаута возвращается сырая транскрипция"))
        vbox.addWidget(box2)

        vbox.addStretch()
        return page

    # ── Вкладка «Уведомления» ─────────────────────────────────────────────────

    def _build_output_tab(self) -> QWidget:
        page = QWidget()
        vbox = QVBoxLayout(page)
        vbox.setContentsMargins(4, 4, 4, 4)

        box, form = _form_group("Windows Toast-уведомления")

        self._w_notify = QCheckBox("Показывать уведомление при готовности текста")
        form.addRow("", self._w_notify)

        self._w_notify_ms = QSpinBox()
        self._w_notify_ms.setRange(1000, 10000)
        self._w_notify_ms.setSuffix(" мс")
        self._w_notify_ms.setSingleStep(500)
        form.addRow("Длительность:", self._w_notify_ms)

        self._w_preview_chars = QSpinBox()
        self._w_preview_chars.setRange(20, 200)
        self._w_preview_chars.setSuffix(" символов")
        form.addRow("Длина превью:", self._w_preview_chars)

        vbox.addWidget(box)
        vbox.addStretch()
        return page

    # ── Загрузка и сбор значений ──────────────────────────────────────────────

    def _load_values(self) -> None:
        """Заполняет виджеты текущими значениями из config."""
        hk  = self._config.get("hotkey", {})
        aud = self._config.get("audio", {})
        stt = self._config.get("transcriber", {})
        ed  = self._config.get("editor", {})
        out = self._config.get("output", {})

        # Запись
        mode = hk.get("mode", "hold")
        self._w_mode.setCurrentIndex(0 if mode == "hold" else 1)
        self._w_hotkey.setText(hk.get("key", "ctrl+shift+space"))
        self._w_min_hold.setValue(hk.get("min_hold_ms", 500))
        self._w_autostop.setValue(hk.get("autostop_sec", 30))
        self._w_max_record.setValue(hk.get("max_record_sec", 300))

        # Аудио
        device = str(aud.get("device", "default"))
        idx = self._w_device.findData(device)
        if idx >= 0:
            self._w_device.setCurrentIndex(idx)

        # STT
        model = stt.get("model", "large-v3-turbo")
        self._w_stt_model.setCurrentIndex(
            max(0, self._w_stt_model.findData(model))
        )
        dev = stt.get("device", "cuda")
        self._w_stt_device.setCurrentIndex(
            max(0, self._w_stt_device.findData(dev))
        )
        ct = stt.get("compute_type", "int8")
        self._w_stt_compute.setCurrentIndex(
            max(0, self._w_stt_compute.findData(ct))
        )
        self._w_beam_size.setValue(stt.get("beam_size", 5))
        lang = stt.get("language") or ""
        self._w_language.setCurrentIndex(
            max(0, self._w_language.findData(lang))
        )

        # Editor
        self._w_editor_enabled.setChecked(ed.get("enabled", True))
        self._w_model_path.setText(ed.get("model_path", "models/Phi-3.5-mini-Q4.gguf"))
        self._w_gpu_layers.setValue(ed.get("gpu_layers", -1))
        self._w_temperature.setValue(ed.get("temperature", 0.1))
        self._w_timeout.setValue(ed.get("timeout_sec", 10))

        # Output
        self._w_notify.setChecked(out.get("notify", True))
        self._w_notify_ms.setValue(out.get("notify_duration_ms", 3000))
        self._w_preview_chars.setValue(out.get("notify_preview_chars", 60))

    def _collect(self) -> dict:
        """Читает текущие значения виджетов в плоский словарь {key_path: value}."""
        lang_val = self._w_language.currentData()
        return {
            "hotkey.mode":          self._w_mode.currentData(),
            "hotkey.key":           self._w_hotkey.text().strip(),
            "hotkey.min_hold_ms":   self._w_min_hold.value(),
            "hotkey.autostop_sec":  self._w_autostop.value(),
            "hotkey.max_record_sec": self._w_max_record.value(),
            "audio.device":         self._w_device.currentData(),
            "transcriber.model":      self._w_stt_model.currentData(),
            "transcriber.device":     self._w_stt_device.currentData(),
            "transcriber.compute_type": self._w_stt_compute.currentData(),
            "transcriber.beam_size":  self._w_beam_size.value(),
            "transcriber.language":   lang_val if lang_val else None,
            "editor.enabled":    self._w_editor_enabled.isChecked(),
            "editor.model_path": self._w_model_path.text().strip(),
            "editor.gpu_layers": self._w_gpu_layers.value(),
            "editor.temperature": self._w_temperature.value(),
            "editor.timeout_sec": self._w_timeout.value(),
            "output.notify":              self._w_notify.isChecked(),
            "output.notify_duration_ms":  self._w_notify_ms.value(),
            "output.notify_preview_chars": self._w_preview_chars.value(),
        }

    def _validate(self, values: dict) -> list[str]:
        errors: list[str] = []
        if not values.get("hotkey.key"):
            errors.append("Горячая клавиша не задана.")
        model_path = values.get("editor.model_path", "")
        if values.get("editor.enabled") and model_path:
            p = Path(model_path)
            if not p.is_absolute():
                p = Path(__file__).parent.parent / p
            if not p.exists():
                errors.append(
                    f"Файл модели Phi не найден:\n{model_path}\n"
                    "Запустите setup.py для скачивания."
                )
        return errors

    # ── Применение изменений ──────────────────────────────────────────────────

    def _apply(self) -> Optional[set]:
        """
        Валидирует, применяет изменения к config in-place, сохраняет.
        Возвращает set изменённых ключей или None при ошибке валидации.
        """
        values = self._collect()
        errors = self._validate(values)
        if errors:
            QMessageBox.warning(self, "Ошибка настроек", "\n".join(errors))
            return None

        # Находим изменённые ключи
        changed: set[str] = set()
        for key_path, new_val in values.items():
            sections = key_path.split(".")
            cur = self._config
            for s in sections[:-1]:
                cur = cur.get(s, {})
            old_val = cur.get(sections[-1])
            if old_val != new_val:
                changed.add(key_path)

        # Обновляем config in-place
        for key_path, new_val in values.items():
            sections = key_path.split(".")
            d = self._config
            for s in sections[:-1]:
                d = d[s]
            d[sections[-1]] = new_val

        # Сохраняем на диск
        if self._cm is not None:
            try:
                self._cm.save()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] settings save: {exc}", file=sys.stderr)

        # Уведомляем о необходимости перезапуска для тяжёлых изменений
        restart_changed = changed & self._RESTART_KEYS
        if restart_changed:
            QMessageBox.information(
                self,
                "VoiceScribe",
                "Изменения настроек моделей применятся после перезапуска приложения.",
            )

        if self._on_applied and changed:
            self._on_applied(changed)

        return changed

    # ── Слоты кнопок ─────────────────────────────────────────────────────────

    def _on_ok(self) -> None:
        if self._apply() is not None:
            self.accept()

    def _on_apply(self) -> None:
        self._apply()

    # ── Вспомогательные действия ─────────────────────────────────────────────

    def _open_hotkey_dialog(self) -> None:
        from ui.tray import HotkeyDialog  # lazy import — нет циклической зависимости
        dlg = HotkeyDialog(self._w_hotkey.text(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_hk = dlg.result_hotkey()
            if new_hk:
                self._w_hotkey.setText(new_hk)

    def _browse_model_path(self) -> None:
        start_dir = str(Path(__file__).parent.parent / "models")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать модель Phi-3.5 Mini",
            start_dir,
            "GGUF файлы (*.gguf);;Все файлы (*)",
        )
        if path:
            # Сохраняем относительный путь если внутри проекта
            try:
                rel = Path(path).relative_to(Path(__file__).parent.parent)
                path = str(rel).replace("\\", "/")
            except ValueError:
                pass
            self._w_model_path.setText(path)
