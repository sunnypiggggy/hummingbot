from __future__ import annotations

import json
from typing import Any, Optional

import requests
from management_bot.risk_display import RichText


class TelegramError(RuntimeError):
    pass


class TelegramAPI:
    def __init__(self, token: str, timeout: float = 35.0):
        self._token = token
        self._base = f"https://api.telegram.org/bot{token}"
        self.timeout = timeout
        self.session = requests.Session()

    def _call(self, method: str, payload: Optional[dict] = None) -> Any:
        try:
            response = self.session.post(f"{self._base}/{method}", data=payload or {}, timeout=self.timeout)
            response.raise_for_status()
            value = response.json()
        except Exception as exc:
            # Never include the URL because it contains the Bot token.
            raise TelegramError(f"Telegram {method} request failed: {type(exc).__name__}") from exc
        if not value.get("ok"):
            raise TelegramError(f"Telegram {method} rejected the request")
        return value.get("result")

    def delete_webhook(self, *, drop_pending_updates: bool) -> Any:
        return self._call("deleteWebhook", {"drop_pending_updates": json.dumps(drop_pending_updates)})

    def set_commands(self, commands: list[dict[str, str]]) -> Any:
        return self._call("setMyCommands", {
            "commands": json.dumps(commands, ensure_ascii=False),
            "scope": json.dumps({"type": "all_private_chats"}),
        })

    def set_commands_menu(self, chat_id: int) -> Any:
        return self._call("setChatMenuButton", {
            "chat_id": str(chat_id),
            "menu_button": json.dumps({"type": "commands"}),
        })

    def get_updates(self, offset: int, timeout: int = 25) -> list[dict]:
        value = self._call("getUpdates", {
            "offset": str(offset),
            "timeout": str(timeout),
            "allowed_updates": json.dumps(["message", "callback_query"]),
        })
        return value if isinstance(value, list) else []

    @staticmethod
    def _markup(rows: Optional[list[list[tuple[str, str]]]]) -> str:
        buttons = []
        for row in rows or []:
            buttons.append([{"text": label, "callback_data": data} for label, data in row])
        return json.dumps({"inline_keyboard": buttons}, ensure_ascii=False)

    def send(self, chat_id: int, text: str, rows: Optional[list[list[tuple[str, str]]]] = None) -> dict:
        return self._call("sendMessage", {
            "chat_id": str(chat_id),
            "text": text[:4096],
            **({"parse_mode": "HTML"} if isinstance(text, RichText) else {}),
            "reply_markup": self._markup(rows),
        })

    def edit(self, chat_id: int, message_id: int, text: str,
             rows: Optional[list[list[tuple[str, str]]]] = None) -> Any:
        return self._call("editMessageText", {
            "chat_id": str(chat_id),
            "message_id": str(message_id),
            "text": text[:4096],
            **({"parse_mode": "HTML"} if isinstance(text, RichText) else {}),
            "reply_markup": self._markup(rows),
        })

    def answer_callback(self, callback_id: str, text: str = "", alert: bool = False) -> Any:
        return self._call("answerCallbackQuery", {
            "callback_query_id": callback_id,
            "text": text[:200],
            "show_alert": json.dumps(alert),
        })

    def send_file(self, chat_id: int, path: str, caption: str = "") -> Any:
        method = "sendPhoto" if path.lower().endswith((".png", ".jpg", ".jpeg")) else "sendDocument"
        field = "photo" if method == "sendPhoto" else "document"
        try:
            with open(path, "rb") as stream:
                response = self.session.post(
                    f"{self._base}/{method}",
                    data={"chat_id": str(chat_id), "caption": caption[:1024]},
                    files={field: stream},
                    timeout=max(self.timeout, 60),
                )
            response.raise_for_status()
            value = response.json()
        except Exception as exc:
            raise TelegramError(f"Telegram {method} request failed: {type(exc).__name__}") from exc
        if not value.get("ok"):
            raise TelegramError(f"Telegram {method} rejected the request")
        return value.get("result")
