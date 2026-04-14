"""
core/config.py — менеджер конфигурации.

Читает и пишет config.yaml через ruamel.yaml (сохраняет комментарии).
Возвращает живой объект-словарь: изменения в нём отражаются при save().

Использование:
    cm = ConfigManager(Path("config.yaml"))
    config = cm.data          # CommentedMap (ведёт себя как dict)
    config["hotkey"]["mode"]  # чтение
    cm.set("hotkey.mode", "toggle")
    cm.save()                 # запись с сохранением комментариев
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigManager:
    """
    Тонкая обёртка вокруг ruamel.yaml.
    Хранит единственный экземпляр данных (CommentedMap);
    все компоненты приложения работают с ним напрямую через .data.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._yaml = self._make_yaml()
        self._data = self._load()

    # ── Публичный API ─────────────────────────────────────────────────────────

    @property
    def data(self):
        """
        Возвращает живой CommentedMap — тот же объект, что хранится внутри.
        Передавайте его в Pipeline / Transcriber / Tray как обычный dict.
        Любые изменения через config["section"]["key"] = value
        немедленно отражаются здесь; после вызова save() они попадут на диск.
        """
        return self._data

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Читает значение по пути вида 'hotkey.mode'.
        Возвращает default если ключ не найден.
        """
        keys = key_path.split(".")
        d = self._data
        for k in keys:
            if not hasattr(d, "__getitem__") or k not in d:
                return default
            d = d[k]
        return d

    def set(self, key_path: str, value: Any) -> None:
        """
        Устанавливает значение по пути вида 'hotkey.mode'.
        Промежуточные секции должны существовать.
        """
        keys = key_path.split(".")
        d = self._data
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    def update_section(self, section: str, values: dict) -> None:
        """Массовое обновление секции: update_section('hotkey', {'mode': 'toggle'})."""
        for key, value in values.items():
            self.set(f"{section}.{key}", value)

    def save(self) -> None:
        """Записывает текущее состояние в config.yaml (с комментариями)."""
        with open(self._path, "w", encoding="utf-8") as f:
            self._yaml.dump(self._data, f)

    def reload(self) -> None:
        """Перечитывает config.yaml с диска (сбрасывает несохранённые изменения)."""
        self._data = self._load()

    # ── Внутренние методы ─────────────────────────────────────────────────────

    @staticmethod
    def _make_yaml():
        try:
            from ruamel.yaml import YAML  # type: ignore
            y = YAML()
            y.preserve_quotes = True
            y.width = 120
            return y
        except ImportError as exc:
            raise RuntimeError(
                "ruamel.yaml не установлен. Запустите: pip install ruamel.yaml"
            ) from exc

    def _load(self):
        if not self._path.exists():
            raise FileNotFoundError(f"config.yaml не найден: {self._path}")
        with open(self._path, encoding="utf-8") as f:
            return self._yaml.load(f)
