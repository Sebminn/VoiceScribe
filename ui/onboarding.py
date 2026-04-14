"""
ui/onboarding.py — экран приветствия (первый запуск).

Показывается один раз при первом запуске (config.app.first_run = true).
При повторных запусках пропускается — приложение уходит сразу в трей.

Структура экрана (снизу вверх — fade-in с задержкой):
    • Анимированный логотип — осциллирующие столбики → "VS"
    • Заголовок + подзаголовок
    • Прогресс-бар загрузки моделей (реальный, из Pipeline)
    • Горячая клавиша [Ctrl] + [Shift] + [Space]
    • Карточки режимов: Hold / Toggle (кликабельны)
    • Кнопка «Начать работу» (активна после загрузки)

Технически:
    QDialog (FramelessWindowHint) + paintEvent для фона с сеткой точек
    QGraphicsOpacityEffect + QPropertyAnimation для fade-in элементов
    Отдельный QThread для загрузки моделей
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import (
    QEasingCurve, QObject, QPoint, QPropertyAnimation,
    QRect, QSequentialAnimationGroup, QSize, Qt, QThread, QTimer,
    pyqtSignal, pyqtSlot,
)
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QGradient, QLinearGradient,
    QPainter, QPen, QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFrame, QGraphicsOpacityEffect,
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)


# ── Цвета ─────────────────────────────────────────────────────────────────────
_BG         = "#0F172A"
_BLUE       = "#2563EB"
_PURPLE     = "#7C3AED"
_WHITE      = "#FFFFFF"
_MUTED      = "rgba(255,255,255,150)"


# ── Загрузчик моделей ─────────────────────────────────────────────────────────

class _Loader(QThread):
    progress = pyqtSignal(str, int)   # сообщение, процент 0-100
    finished = pyqtSignal()
    error    = pyqtSignal(str)

    def __init__(self, pipeline) -> None:
        super().__init__()
        self._pipeline = pipeline

    def run(self) -> None:
        try:
            def _cb(msg: str, frac: float) -> None:
                self.progress.emit(msg, max(0, min(100, int(frac * 100))))
            self._pipeline.load_models(on_progress=_cb)
            self.finished.emit()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


# ── Анимированный логотип (столбики звуковой волны) ───────────────────────────

class _LogoWidget(QWidget):
    """5 вертикальных столбиков, осциллирующих как звуковая волна."""

    BAR_COUNT = 7
    BAR_W     = 10
    BAR_GAP   = 6
    FPS       = 40

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        w = self.BAR_COUNT * self.BAR_W + (self.BAR_COUNT - 1) * self.BAR_GAP
        self.setFixedSize(w + 4, 60)
        self._t     = 0.0
        self._phase = [i * 0.7 for i in range(self.BAR_COUNT)]
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // self.FPS)

    def _tick(self) -> None:
        self._t += 0.08
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        total_w = self.BAR_COUNT * self.BAR_W + (self.BAR_COUNT - 1) * self.BAR_GAP
        x0 = (self.width() - total_w) // 2
        max_h = self.height() - 8

        for i in range(self.BAR_COUNT):
            h_frac = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(self._t + self._phase[i]))
            h = max(6, int(max_h * h_frac))
            x = x0 + i * (self.BAR_W + self.BAR_GAP)
            y = (self.height() - h) // 2

            grad = QLinearGradient(x, y, x, y + h)
            grad.setColorAt(0.0, QColor(_PURPLE))
            grad.setColorAt(1.0, QColor(_BLUE))
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRect(x, y, self.BAR_W, h), 4, 4)

        p.end()


# ── Карточка режима ───────────────────────────────────────────────────────────

class _ModeCard(QFrame):
    """Кликабельная карточка Hold / Toggle с glass-morphism стилем."""

    clicked = pyqtSignal(str)   # 'hold' | 'toggle'

    _STYLE_IDLE = """
        QFrame {{
            background-color: rgba(255,255,255,18);
            border: 2px solid rgba(255,255,255,45);
            border-radius: 14px;
        }}
        QFrame:hover {{
            background-color: rgba(255,255,255,32);
            border-color: rgba(255,255,255,80);
        }}
    """
    _STYLE_ACTIVE = """
        QFrame {{
            background-color: rgba(37,99,235,55);
            border: 2px solid {blue};
            border-radius: 14px;
        }}
    """.format(blue=_BLUE)

    def __init__(self, mode: str, icon: str, title: str, desc: str,
                 parent=None) -> None:
        super().__init__(parent)
        self._mode = mode
        self.setFixedSize(200, 110)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(self._STYLE_IDLE)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(16, 12, 16, 12)
        vbox.setSpacing(6)

        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("font-size: 24px; background: transparent;")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        vbox.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            "color: white; font-size: 14px; font-weight: bold; background: transparent;"
        )
        vbox.addWidget(title_lbl)

        desc_lbl = QLabel(desc)
        desc_lbl.setStyleSheet(
            "color: rgba(255,255,255,160); font-size: 11px; background: transparent;"
        )
        desc_lbl.setWordWrap(True)
        vbox.addWidget(desc_lbl)

    def set_active(self, active: bool) -> None:
        self.setStyleSheet(self._STYLE_ACTIVE if active else self._STYLE_IDLE)

    def mousePressEvent(self, _) -> None:
        self.clicked.emit(self._mode)


# ── Клавиша ───────────────────────────────────────────────────────────────────

def _key_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet("""
        QLabel {
            color: white;
            background-color: rgba(255,255,255,18);
            border: 1px solid rgba(255,255,255,55);
            border-bottom: 3px solid rgba(255,255,255,30);
            border-radius: 6px;
            font-size: 13px;
            font-weight: bold;
            padding: 6px 14px;
        }
    """)
    return lbl


# ── Главный диалог ────────────────────────────────────────────────────────────

class OnboardingDialog(QDialog):
    """
    Экран первого запуска.

    Создаёт и загружает Pipeline внутри себя.
    После закрытия: pipeline доступен через .pipeline (уже загружен),
    config['app']['first_run'] установлен в False.

    Параметры:
        config         — живой CommentedMap из ConfigManager.data
        config_manager — ConfigManager для сохранения изменений
    """

    def __init__(self, config, config_manager=None) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog,
        )
        self._config = config
        self._cm     = config_manager
        self._pipeline_loaded = False

        # Создаём Pipeline (не загружаем — это делает _Loader в фоне)
        from core.pipeline import Pipeline  # noqa: PLC0415
        self._pipeline = Pipeline(config)

        self._loader: Optional[_Loader] = None
        self._card_hold:   _ModeCard
        self._card_toggle: _ModeCard
        self._btn_start:   QPushButton
        self._progress:    QProgressBar
        self._status_lbl:  QLabel

        self._build_ui()
        self._apply_mode_selection(config.get("hotkey", {}).get("mode", "hold"))

    # ── Свойства ─────────────────────────────────────────────────────────────

    @property
    def pipeline(self):
        """Возвращает Pipeline (загружен если pipeline_loaded = True)."""
        return self._pipeline

    @property
    def pipeline_loaded(self) -> bool:
        return self._pipeline_loaded

    # ── Построение UI ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        # Корневой слой — тёмный фон (точки рисуются в paintEvent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Центральная панель (max 560px)
        panel = QWidget()
        panel.setFixedWidth(560)
        panel.setStyleSheet("background: transparent;")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        panel_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._add_logo_section(panel_layout)
        self._add_progress_section(panel_layout)
        self._add_hotkey_section(panel_layout)
        self._add_mode_section(panel_layout)
        self._add_button_section(panel_layout)

        root.addWidget(panel, alignment=Qt.AlignmentFlag.AlignCenter)

        # Запуск анимаций входа
        QTimer.singleShot(100, self._start_entrance_animations)
        # Запуск загрузки моделей
        QTimer.singleShot(400, self._start_loading)

    def _add_logo_section(self, layout: QVBoxLayout) -> None:
        box = QWidget()
        box.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(box)
        vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.setSpacing(10)
        vbox.setContentsMargins(0, 0, 0, 16)

        logo = _LogoWidget()
        logo.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(logo, alignment=Qt.AlignmentFlag.AlignCenter)

        title = QLabel("VoiceScribe")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "color: white; font-size: 38px; font-weight: 800; "
            "letter-spacing: 2px; background: transparent;"
        )
        vbox.addWidget(title)

        sub = QLabel("Голос → Текст.  Локально.  Мгновенно.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(
            "color: rgba(255,255,255,170); font-size: 15px; background: transparent;"
        )
        vbox.addWidget(sub)

        layout.addWidget(box)
        self._logo_section = box

    def _add_progress_section(self, layout: QVBoxLayout) -> None:
        box = QWidget()
        box.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(0, 0, 0, 20)
        vbox.setSpacing(8)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(8)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: rgba(255,255,255,20);
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {_BLUE}, stop:1 {_PURPLE}
                );
                border-radius: 4px;
            }}
        """)
        vbox.addWidget(self._progress)

        self._status_lbl = QLabel("Инициализация...")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet(
            "color: rgba(255,255,255,160); font-size: 13px; background: transparent;"
        )
        vbox.addWidget(self._status_lbl)

        layout.addWidget(box)
        self._progress_section = box

    def _add_hotkey_section(self, layout: QVBoxLayout) -> None:
        box = QWidget()
        box.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(0, 0, 0, 20)
        vbox.setSpacing(8)
        vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hint = QLabel("Горячая клавиша по умолчанию")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(
            "color: rgba(255,255,255,130); font-size: 12px; background: transparent;"
        )
        vbox.addWidget(hint)

        keys_row = QHBoxLayout()
        keys_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        keys_row.setSpacing(8)

        hotkey_str = self._config.get("hotkey", {}).get("key", "ctrl+shift+space")
        parts = [p.strip().capitalize() for p in hotkey_str.split("+")]
        for i, part in enumerate(parts):
            keys_row.addWidget(_key_label(part))
            if i < len(parts) - 1:
                plus = QLabel("+")
                plus.setStyleSheet(
                    "color: rgba(255,255,255,120); font-size: 16px; background: transparent;"
                )
                keys_row.addWidget(plus)

        keys_widget = QWidget()
        keys_widget.setStyleSheet("background: transparent;")
        keys_widget.setLayout(keys_row)
        vbox.addWidget(keys_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(box)
        self._hotkey_section = box

    def _add_mode_section(self, layout: QVBoxLayout) -> None:
        box = QWidget()
        box.setStyleSheet("background: transparent;")
        vbox = QVBoxLayout(box)
        vbox.setContentsMargins(0, 0, 0, 28)
        vbox.setSpacing(10)
        vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl = QLabel("Выберите режим записи")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            "color: rgba(255,255,255,130); font-size: 12px; background: transparent;"
        )
        vbox.addWidget(lbl)

        cards_row = QHBoxLayout()
        cards_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cards_row.setSpacing(16)

        self._card_hold = _ModeCard(
            "hold", "🖱",
            "Удержание (Hold)",
            "Держите клавишу пока говорите. Лучше для коротких фраз.",
        )
        self._card_toggle = _ModeCard(
            "toggle", "🔄",
            "Свободные руки (Toggle)",
            "Нажмите → говорите → нажмите снова. Для длинных диктовок.",
        )
        self._card_hold.clicked.connect(self._on_mode_selected)
        self._card_toggle.clicked.connect(self._on_mode_selected)
        cards_row.addWidget(self._card_hold)
        cards_row.addWidget(self._card_toggle)

        cards_widget = QWidget()
        cards_widget.setStyleSheet("background: transparent;")
        cards_widget.setLayout(cards_row)
        vbox.addWidget(cards_widget, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(box)
        self._mode_section = box

    def _add_button_section(self, layout: QVBoxLayout) -> None:
        self._btn_start = QPushButton("Начать работу")
        self._btn_start.setObjectName("startBtn")
        self._btn_start.setFixedSize(220, 50)
        self._btn_start.setEnabled(False)
        self._btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start.setStyleSheet(f"""
            QPushButton#startBtn {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {_BLUE}, stop:1 {_PURPLE}
                );
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            QPushButton#startBtn:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1D4ED8, stop:1 #6D28D9
                );
            }}
            QPushButton#startBtn:disabled {{
                background: rgba(255,255,255,25);
                color: rgba(255,255,255,80);
            }}
        """)
        self._btn_start.clicked.connect(self._on_start)
        layout.addWidget(self._btn_start, alignment=Qt.AlignmentFlag.AlignCenter)
        self._btn_section = self._btn_start

    # ── Фон с сеткой точек ────────────────────────────────────────────────────

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(_BG))

        dot_color = QColor(255, 255, 255, 18)
        p.setBrush(QBrush(dot_color))
        p.setPen(Qt.PenStyle.NoPen)
        spacing = 32
        for x in range(0, self.width() + spacing, spacing):
            for y in range(0, self.height() + spacing, spacing):
                p.drawEllipse(x - 1, y - 1, 2, 2)
        p.end()

    # ── Анимации входа ────────────────────────────────────────────────────────

    def _start_entrance_animations(self) -> None:
        delays = [
            (self._logo_section,    0),
            (self._progress_section, 300),
            (self._hotkey_section,  550),
            (self._mode_section,    750),
            (self._btn_section,     950),
        ]
        for widget, delay in delays:
            self._fade_in(widget, delay, duration=600)

    def _fade_in(self, widget: QWidget, delay_ms: int, duration: int = 600) -> None:
        effect = QGraphicsOpacityEffect()
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)

        def _start() -> None:
            anim = QPropertyAnimation(effect, b"opacity")
            anim.setDuration(duration)
            anim.setStartValue(0.0)
            anim.setEndValue(1.0)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.start()
            widget._fade_anim = anim  # держим ссылку

        QTimer.singleShot(delay_ms, _start)

    # ── Загрузка моделей ──────────────────────────────────────────────────────

    def _start_loading(self) -> None:
        self._loader = _Loader(self._pipeline)
        self._loader.progress.connect(self._on_progress)
        self._loader.finished.connect(self._on_loading_done)
        self._loader.error.connect(self._on_loading_error)
        self._loader.start()

    @pyqtSlot(str, int)
    def _on_progress(self, message: str, percent: int) -> None:
        self._progress.setValue(percent)
        self._status_lbl.setText(message)

    @pyqtSlot()
    def _on_loading_done(self) -> None:
        self._pipeline_loaded = True
        self._progress.setValue(100)
        self._status_lbl.setText("✓  Все модели загружены — можно начинать!")
        self._status_lbl.setStyleSheet(
            f"color: #4ADE80; font-size: 13px; background: transparent;"
        )
        self._btn_start.setEnabled(True)
        # Лёгкая пульсация кнопки
        self._pulse_button()

    @pyqtSlot(str)
    def _on_loading_error(self, message: str) -> None:
        self._status_lbl.setText(f"⚠  Ошибка: {message}")
        self._status_lbl.setStyleSheet(
            "color: #F87171; font-size: 12px; background: transparent;"
        )
        # Разрешаем закрыть даже при ошибке
        self._btn_start.setEnabled(True)
        self._btn_start.setText("Продолжить без LLM")

    def _pulse_button(self) -> None:
        """Лёгкое мигание кнопки когда модели загружены."""
        effect = QGraphicsOpacityEffect()
        effect.setOpacity(1.0)
        self._btn_start.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(800)
        anim.setStartValue(1.0)
        anim.setEndValue(0.6)
        anim.setEasingCurve(QEasingCurve.Type.SineCurve)
        anim.setLoopCount(3)
        anim.finished.connect(lambda: effect.setOpacity(1.0))
        anim.start()
        self._btn_start._pulse = anim

    # ── Режим ─────────────────────────────────────────────────────────────────

    def _on_mode_selected(self, mode: str) -> None:
        self._config["hotkey"]["mode"] = mode
        self._apply_mode_selection(mode)
        if self._cm is not None:
            try:
                self._cm.save()
            except Exception:  # noqa: BLE001
                pass

    def _apply_mode_selection(self, mode: str) -> None:
        self._card_hold.set_active(mode == "hold")
        self._card_toggle.set_active(mode == "toggle")

    # ── Завершение ────────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        # Помечаем что онбординг пройден
        if "app" in self._config:
            self._config["app"]["first_run"] = False
        if self._cm is not None:
            try:
                self._cm.save()
            except Exception:  # noqa: BLE001
                pass
        self.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            pass  # нельзя закрыть Escape — только через кнопку
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._btn_start.isEnabled():
                self._on_start()
        else:
            super().keyPressEvent(event)
