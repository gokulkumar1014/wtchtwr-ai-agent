from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from functools import lru_cache
from typing import Iterable, Optional, Tuple


Attachment = Tuple[str, bytes, str]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class EmailSettings:
    host: str
    port: int
    username: str | None
    password: str | None
    use_tls: bool
    use_ssl: bool
    sender: str

    @property
    def configured(self) -> bool:
        return bool(self.host and self.sender)


@lru_cache()
def get_settings() -> EmailSettings:
    host = os.getenv("MAIL_HOST", "localhost")
    port = int(os.getenv("MAIL_PORT", "1025"))
    username = os.getenv("MAIL_USERNAME") or None
    password = os.getenv("MAIL_PASSWORD") or None
    use_tls = _bool_env("MAIL_USE_TLS", False)
    use_ssl = _bool_env("MAIL_USE_SSL", False)
    sender = os.getenv("MAIL_FROM", "hope-agent@example.com")

    if use_tls and use_ssl:
        raise RuntimeError("MAIL_USE_TLS and MAIL_USE_SSL cannot both be enabled.")

    return EmailSettings(
        host=host,
        port=port,
        username=username,
        password=password,
        use_tls=use_tls,
        use_ssl=use_ssl,
        sender=sender,
    )


def is_configured() -> bool:
    try:
        settings = get_settings()
    except RuntimeError:
        return False
    return settings.configured


def send_email(
    recipient: str,
    subject: str,
    body: str,
    *,
    attachments: Iterable[Attachment] = (),
    html_body: Optional[str] = None,
) -> None:
    settings = get_settings()
    if not settings.configured:
        raise RuntimeError("Email settings are incomplete. Configure MAIL_HOST and MAIL_FROM to enable email delivery.")

    message = EmailMessage()
    message["From"] = settings.sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body or "")

    html_part = None
    if html_body:
        message.add_alternative(html_body, subtype="html")
        payload = message.get_payload()
        if isinstance(payload, list):
            html_part = payload[-1]

    for filename, content, mimetype in attachments:
        maintype, subtype = ("application", "octet-stream")
        if "/" in mimetype:
            maintype, subtype = mimetype.split("/", 1)
        message.add_attachment(content, maintype=maintype, subtype=subtype, filename=filename)

    try:
        if settings.use_ssl:
            with smtplib.SMTP_SSL(settings.host, settings.port, timeout=15) as smtp:
                if settings.username and settings.password:
                    smtp.login(settings.username, settings.password)
                smtp.send_message(message)
        else:
            with smtplib.SMTP(settings.host, settings.port, timeout=15) as smtp:
                if settings.use_tls:
                    smtp.starttls()
                if settings.username and settings.password:
                    smtp.login(settings.username, settings.password)
                smtp.send_message(message)
    except Exception as exc:  # pragma: no cover - network/SMTP errors
        raise RuntimeError(f"Failed to send email: {exc}") from exc
