"""
VoiceScribe — установщик окружения.

Что делает:
  1. Проверяет версию Python (3.11 / 3.12)
  2. Проверяет NVIDIA-драйвер (>= 572) и CUDA (>= 12.8)
  3. Проверяет / устанавливает CTranslate2 (>= 4.7.1)
  4. Создаёт виртуальное окружение .venv
  5. Устанавливает зависимости из requirements.txt
  6. Устанавливает llama-cpp-python с поддержкой CUDA
  7. Скачивает модели Faster-Whisper и Phi-3.5 Mini

Запуск:
    python setup.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VENV = ROOT / ".venv"
MODELS_DIR = ROOT / "models"
RESOURCES_DIR = ROOT / "resources"

# ── Требования ────────────────────────────────────────────────────────────────
MIN_PYTHON = (3, 11)
MAX_PYTHON = (3, 13)          # llama-cpp-python поддерживает 3.13 начиная с v0.3+
MIN_DRIVER = 572
MIN_CUDA = (12, 8)
MIN_CTRANSLATE2 = (4, 7, 1)

WHISPER_MODEL_REPO = "deepdml/faster-whisper-large-v3-turbo-ct2"
WHISPER_MODEL_DIR  = MODELS_DIR / "faster-whisper-large-v3-turbo"

PHI_REPO      = "bartowski/Phi-3.5-mini-instruct-GGUF"
PHI_FILENAME  = "Phi-3.5-mini-instruct-Q4_K_M.gguf"
PHI_LOCAL     = MODELS_DIR / "Phi-3.5-mini-Q4.gguf"


# ── Утилиты ───────────────────────────────────────────────────────────────────

def info(msg: str)  -> None: print(f"  [INFO]  {msg}")
def ok(msg: str)    -> None: print(f"  [ OK ]  {msg}")
def warn(msg: str)  -> None: print(f"  [WARN]  {msg}")
def error(msg: str) -> None: print(f"  [ERR ]  {msg}")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── Шаг 1: Python ─────────────────────────────────────────────────────────────

def check_python() -> None:
    section("Шаг 1 / 7 — Проверка Python")
    v = sys.version_info
    info(f"Версия: {v.major}.{v.minor}.{v.micro}")

    if v >= (MAX_PYTHON[0], MAX_PYTHON[1] + 1):
        error(
            f"Python {v.major}.{v.minor} не поддерживается llama-cpp-python. "
            f"Используйте Python 3.11 или 3.12."
        )
        sys.exit(1)

    if (v.major, v.minor) < MIN_PYTHON:
        error(f"Минимальная версия Python — 3.11. Текущая: {v.major}.{v.minor}")
        sys.exit(1)

    ok(f"Python {v.major}.{v.minor} — подходит")


# ── Шаг 2: NVIDIA-драйвер и CUDA ──────────────────────────────────────────────

def check_nvidia() -> None:
    section("Шаг 2 / 7 — Проверка NVIDIA-драйвера и CUDA")

    # nvidia-smi
    try:
        result = run(["nvidia-smi", "--query-gpu=driver_version,name",
                      "--format=csv,noheader"])
        line = result.stdout.strip().splitlines()[0]
        driver_str, gpu_name = [x.strip() for x in line.split(",", 1)]
        driver_major = int(driver_str.split(".")[0])
        info(f"GPU: {gpu_name}")
        info(f"Драйвер: {driver_str}")
        if driver_major < MIN_DRIVER:
            error(
                f"Требуется драйвер >= {MIN_DRIVER}.x. "
                f"Скачайте: https://www.nvidia.com/Download/index.aspx"
            )
            sys.exit(1)
        ok(f"Драйвер {driver_str} — OK")
    except (subprocess.CalledProcessError, FileNotFoundError):
        error("nvidia-smi не найден. Установите NVIDIA-драйвер >= 572.x.")
        sys.exit(1)

    # nvcc / CUDA
    try:
        result = run(["nvcc", "--version"])
        m = re.search(r"release (\d+)\.(\d+)", result.stdout)
        if m:
            cuda_major, cuda_minor = int(m.group(1)), int(m.group(2))
            info(f"CUDA: {cuda_major}.{cuda_minor}")
            if (cuda_major, cuda_minor) < MIN_CUDA:
                error(
                    f"Требуется CUDA >= {MIN_CUDA[0]}.{MIN_CUDA[1]}. "
                    f"Скачайте: https://developer.nvidia.com/cuda-downloads"
                )
                sys.exit(1)
            ok(f"CUDA {cuda_major}.{cuda_minor} — OK")
        else:
            warn("Не удалось определить версию CUDA из nvcc. Продолжаем.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        warn("nvcc не найден. Убедитесь, что CUDA Toolkit установлен и добавлен в PATH.")


# ── Шаг 3: Виртуальное окружение ──────────────────────────────────────────────

def create_venv() -> None:
    section("Шаг 3 / 7 — Виртуальное окружение (.venv)")

    if VENV.exists():
        ok(f".venv уже существует: {VENV}")
        return

    info("Создаём .venv ...")
    run([sys.executable, "-m", "venv", str(VENV)])
    ok(f".venv создан: {VENV}")


def venv_python() -> str:
    """Путь к python внутри .venv."""
    if sys.platform == "win32":
        return str(VENV / "Scripts" / "python.exe")
    return str(VENV / "bin" / "python")


def venv_pip() -> str:
    if sys.platform == "win32":
        return str(VENV / "Scripts" / "pip.exe")
    return str(VENV / "bin" / "pip")


# ── Шаг 4: Зависимости из requirements.txt ────────────────────────────────────

def install_requirements() -> None:
    section("Шаг 4 / 7 — Зависимости (requirements.txt)")
    info("Устанавливаем пакеты ... (может занять несколько минут)")

    pip = venv_pip()
    run([pip, "install", "--upgrade", "pip"])

    req_file = str(ROOT / "requirements.txt")
    result = subprocess.run(
        [pip, "install", "-r", req_file],
        text=True,
    )
    if result.returncode != 0:
        error("Ошибка установки зависимостей. Проверьте вывод выше.")
        sys.exit(1)

    ok("Зависимости установлены")


# ── Шаг 5: llama-cpp-python с CUDA ────────────────────────────────────────────

def install_llama_cpp() -> None:
    section("Шаг 5 / 7 — llama-cpp-python (CUDA-сборка)")

    python = venv_python()

    # Проверяем, не установлен ли уже
    check = subprocess.run(
        [python, "-c", "import llama_cpp; print(llama_cpp.__version__)"],
        capture_output=True, text=True
    )
    if check.returncode == 0:
        ok(f"llama-cpp-python уже установлен: v{check.stdout.strip()}")
        return

    info("Собираем llama-cpp-python с поддержкой CUDA ...")
    info("Это может занять 5–15 минут (компиляция C++).")

    env = os.environ.copy()
    env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
    env["FORCE_CMAKE"] = "1"

    result = subprocess.run(
        [venv_pip(), "install", "llama-cpp-python>=0.3.4", "--no-cache-dir"],
        env=env,
        text=True,
    )
    if result.returncode != 0:
        warn(
            "Не удалось скомпилировать llama-cpp-python с CUDA. "
            "Пробуем предсобранный wheel для CUDA 12 ..."
        )
        # Предсобранные колёса от abetlen
        wheel_url = (
            "https://github.com/abetlen/llama-cpp-python/releases/download/"
            "v0.3.4/llama_cpp_python-0.3.4-cp311-cp311-win_amd64.whl"
        )
        fallback = subprocess.run(
            [venv_pip(), "install", wheel_url],
            text=True,
        )
        if fallback.returncode != 0:
            error(
                "Не удалось установить llama-cpp-python. "
                "Установите вручную: https://github.com/abetlen/llama-cpp-python/releases"
            )
            sys.exit(1)

    ok("llama-cpp-python установлен с CUDA")


# ── Шаг 6: Проверка CTranslate2 ───────────────────────────────────────────────

def check_ctranslate2() -> None:
    section("Шаг 6 / 7 — Проверка CTranslate2")
    python = venv_python()

    result = subprocess.run(
        [python, "-c",
         "import ctranslate2; v=ctranslate2.__version__; print(v)"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        error("CTranslate2 не установлен. Убедитесь, что requirements.txt применён.")
        sys.exit(1)

    ver_str = result.stdout.strip()
    parts = [int(x) for x in ver_str.split(".")[:3]]
    ver = tuple(parts)

    info(f"CTranslate2: {ver_str}")
    if ver < MIN_CTRANSLATE2:
        error(
            f"Требуется CTranslate2 >= {'.'.join(str(x) for x in MIN_CTRANSLATE2)}. "
            f"Текущая: {ver_str}. Обновите: pip install --upgrade ctranslate2"
        )
        sys.exit(1)

    ok(f"CTranslate2 {ver_str} — OK")


# ── Шаг 7: Скачивание моделей ─────────────────────────────────────────────────

def download_models() -> None:
    section("Шаг 7 / 7 — Загрузка моделей")

    python = venv_python()
    MODELS_DIR.mkdir(exist_ok=True)

    # Faster-Whisper large-v3-turbo
    if WHISPER_MODEL_DIR.exists() and any(WHISPER_MODEL_DIR.iterdir()):
        ok(f"Faster-Whisper уже скачан: {WHISPER_MODEL_DIR}")
    else:
        info(f"Скачиваем Faster-Whisper large-v3-turbo (~1.6 ГБ) ...")
        info(f"Источник: {WHISPER_MODEL_REPO}")
        download_script = f"""
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="{WHISPER_MODEL_REPO}",
    local_dir="{WHISPER_MODEL_DIR.as_posix()}",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)
print("DONE")
"""
        result = subprocess.run([python, "-c", download_script], text=True)
        if result.returncode != 0:
            error("Ошибка загрузки Faster-Whisper. Проверьте интернет-соединение.")
            sys.exit(1)
        ok("Faster-Whisper large-v3-turbo загружен")

    # Phi-3.5 Mini Q4_K_M
    if PHI_LOCAL.exists():
        ok(f"Phi-3.5 Mini уже скачан: {PHI_LOCAL}")
    else:
        info(f"Скачиваем Phi-3.5 Mini Q4_K_M (~2.4 ГБ) ...")
        info(f"Источник: {PHI_REPO} / {PHI_FILENAME}")
        download_script = f"""
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="{PHI_REPO}",
    filename="{PHI_FILENAME}",
    local_dir="{MODELS_DIR.as_posix()}",
)
import shutil, pathlib
dest = pathlib.Path("{PHI_LOCAL.as_posix()}")
if pathlib.Path(path).resolve() != dest.resolve():
    shutil.copy2(path, dest)
print("DONE", path)
"""
        result = subprocess.run([python, "-c", download_script], text=True)
        if result.returncode != 0:
            error("Ошибка загрузки Phi-3.5 Mini. Проверьте интернет-соединение.")
            sys.exit(1)
        ok(f"Phi-3.5 Mini Q4_K_M загружен → {PHI_LOCAL.name}")


# ── Создание папок ────────────────────────────────────────────────────────────

def create_structure() -> None:
    """Создаёт пустые папки и __init__.py, если их нет."""
    dirs = [
        ROOT / "core",
        ROOT / "ui",
        ROOT / "models",
        ROOT / "resources",
    ]
    for d in dirs:
        d.mkdir(exist_ok=True)

    for pkg in [ROOT / "core", ROOT / "ui"]:
        init = pkg / "__init__.py"
        if not init.exists():
            init.write_text("# VoiceScribe package\n")

    # .gitkeep для пустых папок
    for folder in [ROOT / "models", ROOT / "resources"]:
        keep = folder / ".gitkeep"
        if not keep.exists():
            keep.write_text("")


# ── Итог ──────────────────────────────────────────────────────────────────────

def print_summary() -> None:
    python = venv_python()
    print(f"""
{'═' * 60}
  VoiceScribe — установка завершена!
{'═' * 60}

  Запуск приложения:
    {python} main.py

  Консольный тест аудио + VAD:
    {python} -m core.audio

  Консольный тест транскрипции:
    {python} -m core.transcriber

  Консольный тест пайплайна:
    {python} -m core.pipeline

{'═' * 60}
""")


# ── Точка входа ───────────────────────────────────────────────────────────────

def main() -> None:
    print("""
╔══════════════════════════════════════════════════════════╗
║           VoiceScribe — Установка окружения              ║
║   Офлайн голосовой ввод для Windows (RTX + CUDA 12.8)    ║
╚══════════════════════════════════════════════════════════╝
""")
    check_python()
    check_nvidia()
    create_structure()
    create_venv()
    install_requirements()
    install_llama_cpp()
    check_ctranslate2()
    download_models()
    print_summary()


if __name__ == "__main__":
    main()
