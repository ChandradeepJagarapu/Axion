import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional
from uuid import uuid4


REMINDER_FILE = Path(__file__).parent / "reminders.json"


class ReminderManager:
    def __init__(
        self,
        notify_callback: Optional[Callable[[str], None]] = None,
    ):
        self.reminders: List[Dict] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.notify_callback = notify_callback

        self.load()

        self.worker_thread = threading.Thread(
            target=self._worker,
            daemon=True,
        )

        self.worker_thread.start()

    # ---------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------

    def load(self) -> None:
        if not REMINDER_FILE.exists():
            self.reminders = []
            return

        try:
            with open(
                REMINDER_FILE,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if isinstance(data, list):
                self.reminders = data
            else:
                self.reminders = []

        except (
            json.JSONDecodeError,
            OSError,
        ):
            self.reminders = []

    def save(self) -> None:
        with open(
            REMINDER_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.reminders,
                file,
                indent=2,
                ensure_ascii=False,
            )

    # ---------------------------------------------------------
    # Add reminder
    # ---------------------------------------------------------

    def add_reminder(
        self,
        message: str,
        remind_at: datetime,
    ) -> Dict:
        reminder = {
            "id": str(uuid4()),
            "message": message,
            "remind_at": remind_at.isoformat(),
            "created_at": datetime.now().isoformat(),
            "completed": False,
        }

        with self.lock:
            self.reminders.append(reminder)
            self.save()

        return reminder

    # ---------------------------------------------------------
    # Get active reminders
    # ---------------------------------------------------------

    def get_active_reminders(self) -> List[Dict]:
        with self.lock:
            return [
                reminder.copy()
                for reminder in self.reminders
                if not reminder.get("completed", False)
            ]

    # ---------------------------------------------------------
    # Remove reminder
    # ---------------------------------------------------------

    def remove_reminder(self, reminder_id: str) -> bool:
        with self.lock:
            for reminder in self.reminders:
                if reminder.get("id") == reminder_id:
                    reminder["completed"] = True
                    self.save()
                    return True

        return False

    # ---------------------------------------------------------
    # Background reminder worker
    # ---------------------------------------------------------

    def _worker(self) -> None:
        while not self.stop_event.is_set():

            now = datetime.now()

            due_reminders = []

            with self.lock:
                for reminder in self.reminders:

                    if reminder.get("completed", False):
                        continue

                    try:
                        remind_at = datetime.fromisoformat(
                            reminder["remind_at"]
                        )
                    except (
                        ValueError,
                        KeyError,
                    ):
                        continue

                    if now >= remind_at:
                        reminder["completed"] = True
                        due_reminders.append(
                            reminder.copy()
                        )

                if due_reminders:
                    self.save()

            # Notify outside the lock.
            for reminder in due_reminders:
                self._notify(reminder)

            # Check frequently so reminders fire close to their time.
            self.stop_event.wait(1)

    # ---------------------------------------------------------
    # Notification
    # ---------------------------------------------------------

    def _notify(self, reminder: Dict) -> None:
        message = (
            "Reminder: "
            + reminder.get(
                "message",
                "You have a reminder.",
            )
        )

        if self.notify_callback:
            try:
                self.notify_callback(message)
            except Exception as error:
                print(
                    f"\nAXION reminder error: {error}"
                )

    # ---------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------

    def stop(self) -> None:
        self.stop_event.set()