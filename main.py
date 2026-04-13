"""
VoiceScribe — точка входа.

Запуск:  python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _load_config() -> dict:
    import yaml  # type: ignore
    p = Path(__file__).parent / "config.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    import os
    # Подавляем лишний вывод llama-cpp до создания QApplication
    os.environ.setdefault("LLAMA_LOG_LEVEL", "3")

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt

    app = QApplication(sys.argv)
    app.setApplicationName("VoiceScribe")
    app.setApplicationDisplayName("VoiceScribe")
    # Трей-приложение: не завершаться при закрытии любого окна
    app.setQuitOnLastWindowClosed(False)

    config = _load_config()

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
    #     dlg = OnboardingDialog(config)
    #     dlg.exec()

    # ── Трей ─────────────────────────────────────────────────────────────────
    from ui.tray import VoiceScribeTray
    tray = VoiceScribeTray(app, config)
    tray.setup()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
