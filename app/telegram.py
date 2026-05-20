from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

_API = "https://api.telegram.org"


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, timeout: float = 10.0) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout = timeout
        self._session = requests.Session()

    def send(self, text: str, max_retries: int = 3) -> bool:
        url = f"{_API}/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True}
        return self._post_json(url, payload, max_retries=max_retries)

    def send_photo(
        self,
        image_bytes: bytes,
        caption: str,
        *,
        filename: str = "snapshot.jpg",
        max_retries: int = 3,
    ) -> bool:
        url = f"{_API}/bot{self.bot_token}/sendPhoto"
        data = {"chat_id": self.chat_id, "caption": caption}
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                r = self._session.post(
                    url,
                    data=data,
                    files={"photo": (filename, image_bytes, "image/jpeg")},
                    timeout=self.timeout,
                )
                if r.status_code == 200:
                    return True
                log.warning("telegram %s: %s", r.status_code, r.text[:200])
                if r.status_code == 429:
                    try:
                        delay = float(r.json().get("parameters", {}).get("retry_after", delay))
                    except Exception:
                        pass
            except requests.RequestException as e:
                log.warning("telegram request error (attempt %d): %s", attempt, e)
            if attempt < max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 30)
        return False

    def _post_json(self, url: str, payload: dict, *, max_retries: int) -> bool:
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                r = self._session.post(url, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    return True
                log.warning("telegram %s: %s", r.status_code, r.text[:200])
                # 429 → respect retry_after
                if r.status_code == 429:
                    try:
                        delay = float(r.json().get("parameters", {}).get("retry_after", delay))
                    except Exception:
                        pass
            except requests.RequestException as e:
                log.warning("telegram request error (attempt %d): %s", attempt, e)
            if attempt < max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 30)
        return False
