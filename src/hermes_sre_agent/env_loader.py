"""不依赖第三方库的 .env 读取器。"""

import os
import re
from pathlib import Path
from typing import Optional


_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_dotenv(path: Optional[Path] = None) -> Path:
    """读取 .env，但绝不覆盖已经由操作系统设置的环境变量。"""
    env_path = path or Path.cwd() / ".env"
    if not env_path.is_file():
        return env_path
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not _KEY_PATTERN.fullmatch(key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return env_path
