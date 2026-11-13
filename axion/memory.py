import json
from pathlib import Path
from typing import Any, Dict


MEMORY_FILE = Path(__file__).resolve().parent / "memory.json"


class Memory:
    def __init__(self, file_path: Path = None):
        self.file_path = Path(file_path) if file_path else MEMORY_FILE
        self.data: Dict[str, Any] = {}
        self.load()

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def update(self, key: str, value: Any) -> None:
        """Set or replace one memory value and persist it."""
        self.set(key, value)

    def get_all(self) -> Dict[str, Any]:
        return dict(self.data)

    def delete(self, key: str) -> bool:
        if key not in self.data:
            return False
        del self.data[key]
        self.save()
        return True

    def clear(self) -> None:
        self.data.clear()
        self.save()

    def save(self, key: str = None, value: Any = None) -> None:
        # Keep save() as the existing persistence operation while also
        # supporting save(key, value) as a convenient public write method.
        if key is not None:
            self.data[key] = value

        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_file = self.file_path.with_suffix(
            ".tmp"
        )

        with open(
            temporary_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.data,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.flush()

        temporary_file.replace(self.file_path)

    def load(self) -> None:
        if not self.file_path.exists():
            self.data = {}
            return

        try:
            with open(
                self.file_path,
                "r",
                encoding="utf-8",
            ) as file:
                loaded = json.load(file)

            if isinstance(loaded, dict):
                self.data = loaded
            else:
                self.data = {}

        except (
            json.JSONDecodeError,
            OSError,
        ) as error:
            print(
                f"AXION memory warning: "
                f"Could not load memory: {error}"
            )
            self.data = {}
