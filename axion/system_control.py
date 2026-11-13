import re
import shutil
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


class SystemAction(str, Enum):
    OPEN_WEBSITE = "open_website"
    OPEN_APPLICATION = "open_application"
    OPEN_FOLDER = "open_folder"
    SYSTEM_STATUS = "system_status"
    CLARIFY = "clarify"


class SystemController:
    """Perform a small set of fixed, allowlisted local actions."""

    WEBSITES = {
        "youtube": "https://www.youtube.com",
        "github": "https://github.com",
        "google": "https://www.google.com",
        "chatgpt": "https://chatgpt.com",
        "linkedin": "https://www.linkedin.com/feed/",
        "whatsapp": "https://www.whatsapp.com",
    }

    APP_ALIASES = {
        "vs code": "vscode",
        "visual studio code": "vscode",
        "code": "vscode",
        "file explorer": "explorer",
        "explorer": "explorer",
        "notepad": "notepad",
        "windows terminal": "windows_terminal",
        "linkedin": "linkedIn",
        "whatsapp": "WhatsApp",
        "debian": "Debian",
        "Calculator": "Calculator"
    }

    def __init__(
        self,
        popen: Callable = subprocess.Popen,
        run: Callable = subprocess.run,
        home: Optional[Path] = None,
        axion_dir: Optional[Path] = None,
        windows_root: Path = Path("/mnt/c/Windows"),
        code_executable: Optional[str] = None,
        windows_terminal_executable: Optional[str] = None,
    ):
        self._popen = popen
        self._run = run
        self.home = Path(home or Path.home()).resolve()
        self.axion_dir = Path(
            axion_dir or Path(__file__).resolve().parent.parent
        ).resolve()
        self.windows_root = Path(windows_root)
        self.code_executable = code_executable or shutil.which("code")
        self.windows_terminal_executable = (
            windows_terminal_executable or shutil.which("wt.exe")
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    @staticmethod
    def _has(text: str, phrase: str) -> bool:
        return bool(re.search(
            r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])",
            text,
        ))

    def classify_action(self, user_input: str) -> SystemAction:
        text = self._normalize(user_input)

        if re.search(r"\b(cpu|processor|ram|memory usage|disk usage|system status)\b", text):
            return SystemAction.SYSTEM_STATUS

        if re.search(
            r"^(?:(?:please|could you|can you|would you)\s+)?"
            r"(?:axion\s+)?(?:open(?:\s+up)?|launch|start)\b",
            text,
        ):
            if re.search(r"\b(folder|directory|home|axion)\b", text):
                return SystemAction.OPEN_FOLDER
            if re.search(r"\b(website|site|url)\b", text):
                return SystemAction.OPEN_WEBSITE
            if re.search(
                r"(?:https?://|www\.|\b[a-z0-9-]+\.(?:com|org|net|io|dev|ai|co)\b)",
                text,
            ):
                return SystemAction.OPEN_WEBSITE
            if any(self._has(text, name) for name in self.WEBSITES):
                return SystemAction.OPEN_WEBSITE
            if any(self._has(text, name) for name in self.APP_ALIASES):
                return SystemAction.OPEN_APPLICATION
            if re.search(r"\b(application|app|program)\b", text):
                return SystemAction.OPEN_APPLICATION
            if re.search(r"\b(it|that|this)\b", text):
                return SystemAction.CLARIFY
            return SystemAction.CLARIFY

        if re.search(r"\b(folder|directory)\b", text) and re.search(
            r"\b(axion|home|open|show)\b", text
        ):
            return SystemAction.OPEN_FOLDER

        return SystemAction.CLARIFY

    def _windows_executable(self, name: str) -> Optional[str]:
        candidates = {
            "explorer": self.windows_root / "explorer.exe",
            "notepad": self.windows_root / "notepad.exe",
            "calculator":self.windows_root / "calculator.exe",
        }
        path = candidates.get(name)
        return str(path) if path is not None and path.is_file() else None

    def _open(self, arguments: list[str], cwd: Optional[Path] = None) -> None:
        self._popen(
            arguments,
            cwd=str(cwd) if cwd else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )

    def open_website(self, name: str) -> str:
        key = self._normalize(name)
        url = self.WEBSITES.get(key)
        if url is None:
            return f"That website isn't supported yet. I can open {', '.join(self.WEBSITES)}."

        explorer = self._windows_executable("explorer")
        if explorer is None:
            return "I can't open websites from this environment because File Explorer is unavailable."
        self._open([explorer, url])
        return f"Opening {key.title()}."

    def open_application(self, name: str) -> str:
        key = self._normalize(name)
        app = self.APP_ALIASES.get(key)
        if app == "vscode":
            if not self.code_executable or not Path(self.code_executable).is_file():
                return "VS Code isn't available in this environment."
            self._open([self.code_executable, str(self.axion_dir)], self.axion_dir)
            return "Opening VS Code with the AXION folder."

        if app in ("explorer", "notepad","calculator"):
            executable = self._windows_executable(app)
            if executable is None:
                return f"{name} isn't available in this environment."
            self._open([executable])
            return f"Opening {name}."

        if app == "windows_terminal":
            if not self.windows_terminal_executable:
                return "Windows Terminal isn't available in this environment."
            self._open([self.windows_terminal_executable])
            return "Opening Windows Terminal."
        return f"That application isn't supported. I can open VS Code, File Explorer, or Notepad."

    def open_folder(self, name: str) -> str:
        key = self._normalize(name)
        if "axion" in key:
            folder = self.axion_dir
            label = "AXION folder"
        elif "home" in key:
            folder = self.home
            label = "home folder"
        else:
            return "I can open the AXION folder or your home folder. Which one do you want?"

        if not folder.is_dir():
            return f"The {label} isn't available at its expected location."

        explorer = self._windows_executable("explorer")
        if explorer is None:
            return "File Explorer isn't available in this environment."

        try:
            converted = self._run(
                ["wslpath", "-w", str(folder)],
                check=True,
                capture_output=True,
                text=True,
                shell=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "I couldn't convert that WSL folder path for Windows Explorer."

        self._open([explorer, converted])
        return f"Opening your {label}."

    @staticmethod
    def _cpu_times() -> tuple[int, int]:
        with open("/proc/stat", "r", encoding="ascii") as stat_file:
            fields = stat_file.readline().split()
        if not fields or fields[0] != "cpu":
            raise OSError("CPU counters are unavailable")
        values = [int(value) for value in fields[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    def get_system_status(self) -> str:
        try:
            total_before, idle_before = self._cpu_times()
            time.sleep(0.1)
            total_after, idle_after = self._cpu_times()
            total_delta = total_after - total_before
            cpu = (
                100.0 * (1 - (idle_after - idle_before) / total_delta)
                if total_delta > 0 else 0.0
            )

            memory = {}
            with open("/proc/meminfo", "r", encoding="ascii") as meminfo:
                for line in meminfo:
                    name, value, *_ = line.split()
                    if name in ("MemTotal:", "MemAvailable:"):
                        memory[name] = int(value) * 1024
            total_ram = memory["MemTotal:"]
            available_ram = memory["MemAvailable:"]
            used_ram = total_ram - available_ram

            disk = shutil.disk_usage(self.home)
            gib = 1024 ** 3
            return (
                f"CPU usage is about {cpu:.0f}%. "
                f"RAM usage is about {used_ram / gib:.1f} GB of "
                f"{total_ram / gib:.1f} GB, with {available_ram / gib:.1f} GB available. "
                f"The home filesystem has {disk.free / gib:.1f} GB free "
                f"of {disk.total / gib:.1f} GB."
            )
        except (OSError, KeyError, ValueError, ZeroDivisionError) as error:
            return f"I couldn't read system status: {error}."

    def handle(self, user_input: str) -> str:
        action = self.classify_action(user_input)
        text = self._normalize(user_input)

        if action is SystemAction.SYSTEM_STATUS:
            return self.get_system_status()

        if action is SystemAction.OPEN_FOLDER:
            return self.open_folder(text)

        if action is SystemAction.OPEN_WEBSITE:
            for name in self.WEBSITES:
                if self._has(text, name):
                    return self.open_website(name)
            if re.search(
                r"(?:https?://|www\.|\b[a-z0-9-]+\.(?:com|org|net|io|dev|ai|co)\b)",
                text,
            ):
                return f"That website isn't supported. I can open {', '.join(self.WEBSITES)}."
            return "Which supported website should I open? I can open YouTube, GitHub, Google, or ChatGPT."

        if action is SystemAction.OPEN_APPLICATION:
            for alias in self.APP_ALIASES:
                if self._has(text, alias):
                    return self.open_application(alias)
            return "Which application should I open? I can open VS Code, File Explorer, or Notepad."

        return "What would you like me to open? I can open an allowlisted website, application, or folder."
