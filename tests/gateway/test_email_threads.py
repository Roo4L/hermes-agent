"""Tests for email threads: thread roots, subject, the HTML part, per-thread
sessions and cold-start catch-up.

Everything here is offline: SMTP and IMAP are mocks, and addresses use test.com
or example.com. This module holds the shared fixtures; the thread tests build
on them.
"""

import asyncio
import json
import os
import unittest
from email.mime.text import MIMEText
from pathlib import Path
from unittest.mock import MagicMock, patch


ENV = {
    "EMAIL_ADDRESS": "hermes@test.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.test.com",
    "EMAIL_SMTP_HOST": "smtp.test.com",
}


def make_adapter(**extra):
    """Build an EmailAdapter with mocked env vars and ``extra`` platform settings."""
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, ENV):
        from plugins.platforms.email.adapter import EmailAdapter

        adapter = EmailAdapter(PlatformConfig(enabled=True, extra=dict(extra)))
    return adapter


def make_inbound(
    *,
    sender="User <user@test.com>",
    subject="Hello",
    body="Hi",
    message_id="<in-1@test.com>",
    in_reply_to=None,
    references=None,
) -> bytes:
    """Raw bytes of an inbound plain-text mail; threading headers only when given."""
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = sender
    msg["To"] = "hermes@test.com"
    msg["Subject"] = subject
    msg["Message-ID"] = message_id
    if in_reply_to is not None:
        msg["In-Reply-To"] = in_reply_to
    if references is not None:
        msg["References"] = references
    return msg.as_bytes()


def _as_bytes(value):
    if isinstance(value, bytes):
        return value
    return str(value).encode()


def fake_imap(uids_all=b"", uids_unseen=b"", messages=None, uidvalidity=b"42"):
    """A MagicMock IMAP4_SSL connection.

    ``uid("search", None, "ALL")`` returns *uids_all*, ``uid("search", None,
    "UNSEEN")`` returns *uids_unseen*, and ``uid("fetch", uid, ...)`` returns the
    raw bytes in *messages* (keys are bytes like ``b"4"``). Anything else is NO.
    ``response("UIDVALIDITY")`` gives *uidvalidity* (``[None]`` when it is None).
    """
    messages = dict(messages or {})
    imap = MagicMock()

    def uid_handler(command, *args):
        command = str(command).lower()
        if command == "search":
            criteria = " ".join(str(a) for a in args if a is not None).upper()
            if "UNSEEN" in criteria:
                return ("OK", [uids_unseen])
            if "ALL" in criteria:
                return ("OK", [uids_all])
            return ("NO", [])
        if command == "fetch" and args:
            uid = _as_bytes(args[0])
            if uid in messages:
                return ("OK", [(uid, messages[uid])])
        return ("NO", [])

    imap.uid.side_effect = uid_handler
    imap.response.return_value = ("UIDVALIDITY", [uidvalidity])
    imap.select.return_value = ("OK", [b"1"])
    imap.login.return_value = ("OK", [b""])
    imap.logout.return_value = ("OK", [b""])
    return imap


def sent_message(mock_smtp_cls):
    """The message object passed to ``send_message`` on a patched ``smtplib.SMTP``."""
    return mock_smtp_cls.return_value.send_message.call_args[0][0]


class DispatchCase(unittest.TestCase):
    """Base for tests that drive ``_dispatch_message`` / the poll path.

    Opts into allow-all and From-header trust (the dispatch mechanics are under
    test, not the authorization gate), and clears the class-level seen-UID
    snapshot so one test's connect never leaks into the next.
    """

    _ENV_FLAGS = {"EMAIL_ALLOW_ALL_USERS": "true", "EMAIL_TRUST_FROM_HEADER": "true"}

    def setUp(self):
        from plugins.platforms.email.adapter import EmailAdapter

        self._prev_env = {name: os.environ.get(name) for name in self._ENV_FLAGS}
        os.environ.update(self._ENV_FLAGS)
        self._prev_snapshot = dict(EmailAdapter._seen_uids_snapshot)
        EmailAdapter._seen_uids_snapshot.clear()

    def tearDown(self):
        from plugins.platforms.email.adapter import EmailAdapter

        for name, value in self._prev_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        EmailAdapter._seen_uids_snapshot.clear()
        EmailAdapter._seen_uids_snapshot.update(self._prev_snapshot)

    def capture(self, adapter):
        """Replace ``adapter.handle_message`` with a recorder; returns the event list."""
        events = []

        async def _capture(event):
            events.append(event)

        adapter.handle_message = _capture
        return events


class TestScaffold(unittest.TestCase):
    def test_scaffold_adapter_store_dir_is_sandboxed(self):
        """The thread store's directory stays inside the per-test HERMES_HOME."""
        from plugins.plugin_storage import plugin_data_dir

        adapter = make_adapter()
        self.assertIsNotNone(adapter)
        home = Path(os.environ["HERMES_HOME"]).resolve()
        self.assertIn(home, plugin_data_dir("email").resolve().parents)


if __name__ == "__main__":
    unittest.main()
