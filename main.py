"""
VoiceScribe — точка входа.

Запуск:  python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    import os
    os.environ.setdefault("LLAMA_LOG_LEVEL", "3")

    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("VoiceScribe")
    app.setApplicationDisplayName("VoiceScribe")
    app.setQuitOnLastWindowClosed(False)

    # ── Конфигурация ──────────────────────────────────────────────────────────
    from core.config import ConfigManager
    cm = ConfigManager(Path(__file__).parent / "config.yaml")
    config = cm.data  # живой CommentedMap — все компоненты работают с ним

    # ── Проверка системного трея ──────────────────────────────────────────────
    from PyQt6.QtWidgets import QSystemTrayIcon
    if not QSystemTrayIcon.isSystemTrayAvailable():
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(
            None,
            "VoiceScribe",
            "Системный трей недоступен. Убедитесь, что панель задач Windows активна.",
        )
        sys.exit(1)

    # ── TODO (Этап 8): onboarding при первом запуске ──────────────────────────
    # if config.get("app", {}).get("first_run", True):
    #     from ui.onboarding import OnboardingDialog
    #     dlg = OnboardingDialog(config, cm)
    #     dlg.exec()

    # ── Трей ─────────────────────────────────────────────────────────────────
    from ui.tray import VoiceScribeTray
    tray = VoiceScribeTray(app, config, config_manager=cm)
    tray.setup()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
