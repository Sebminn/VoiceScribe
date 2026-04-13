# VoiceScribe — Claude Code Instructions

## Проект
Локальный голосовой ассистент для Windows. Офлайн STT + LLM постобработка.
Данные не покидают компьютер. Цель: нажал горячую клавишу → говоришь → текст в буфере обмена.

## Целевая система
- OS: Windows 11
- GPU: NVIDIA RTX 5070 Ti, 12 ГБ VRAM
- CUDA: 12.8 / Driver: >= 572.x
- Python: 3.11 или 3.12 (не 3.13 — не поддерживается llama-cpp)
- CTranslate2: >= 4.7.1

## Технический стек
| Роль | Технология |
|------|-----------|
| Захват звука | sounddevice + numpy |
| VAD | Silero VAD v4 (CPU) |
| STT | Faster-Whisper large-v3-turbo INT8 (~1.5 ГБ VRAM) |
| LLM постобработка | Phi-3.5 Mini Q4_K_M via llama-cpp (~2.2 ГБ VRAM) |
| Горячие клавиши | pynput |
| UI / Трей | PyQt6 |
| Буфер обмена | pyperclip |
| Уведомления | plyer (Windows Toast) |

## Структура проекта
```
voicescribe/
├── main.py
├── config.yaml
├── requirements.txt
├── setup.py
├── core/
│   ├── audio.py        # Захват PCM 16kHz mono
│   ├── vad.py          # Silero VAD — детекция речи
│   ├── transcriber.py  # Faster-Whisper STT
│   ├── editor.py       # Phi-3.5 Mini постобработка
│   └── pipeline.py     # Оркестратор пайплайна
├── ui/
│   ├── tray.py         # Системный трей, иконки, меню
│   ├── onboarding.py   # Экран приветствия (первый запуск)
│   ├── overlay.py      # Индикатор уровня громкости
│   └── settings.py     # Окно настроек
├── models/             # Модели (скачиваются при первом запуске)
└── resources/          # Иконки трея (5 состояний)
```

## Команды запуска
```bash
# Первый запуск — установка зависимостей и скачивание моделей
python setup.py

# Запуск приложения
python main.py

# Консольный тест аудио + VAD
python -m core.audio

# Консольный тест транскрипции
python -m core.transcriber

# Консольный тест пайплайна (без UI)
python -m core.pipeline
```

## Правила кода
- `audio.sample_rate` = 16000 — не менять, Whisper требует 16kHz
- `audio.channels` = 1 — только моно
- Все модели загружаются один раз при старте приложения, не при каждом запросе
- Тяжёлые операции (загрузка моделей, транскрипция) — только в отдельных потоках, не в UI thread
- Конфиг читается из `config.yaml`, изменения применяются через `ConfigManager`

## Запреты
- Не менять `sample_rate` и `channels` в audio — сломает Whisper
- Не блокировать UI thread — все модели и пайплайн работают в QThread / threading
- Не использовать Python 3.13 — llama-cpp не поддерживает

## План MVP (этапы)
1. **Окружение** — setup.py, requirements.txt, config.yaml, структура папок
2. **Аудио + VAD** — audio.py, vad.py, консольный тест
3. **STT** — transcriber.py, Faster-Whisper
4. **LLM** — editor.py, Phi-3.5 Mini
5. **Пайплайн** — pipeline.py, консольный MVP
6. **UI / Трей** — tray.py, иконки, меню, горячие клавиши
7. **Настройки** — settings.py, окно настроек
8. **Onboarding** — onboarding.py, анимация, прогресс загрузки
