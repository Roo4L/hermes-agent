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


class TestSubjectRule(unittest.TestCase):
    """MAIL-01: a leading ``Subject:`` line names the mail, else the job name, else "Hermes Agent"."""

    def test_subject_from_leading_subject_line(self):
        from plugins.platforms.email.adapter import _split_subject

        self.assertEqual(_split_subject("Subject: Weekly report — Nov 9\n\n# Body\n- a", "Fallback"),
                         ("Weekly report — Nov 9", "# Body\n- a"))
        self.assertEqual(_split_subject("\n\nSubject: Late start\nbody", "F"), ("Late start", "body"))
        self.assertEqual(_split_subject("subject: lower case\nbody", "F"), ("lower case", "body"))

    def test_subject_after_cron_failure_marker(self):
        from plugins.platforms.email.adapter import _split_subject

        self.assertEqual(_split_subject("[CRON_FAILURE]\nSubject: Job broke\nDetails", "F"), ("Job broke", "Details"))

    def test_subject_not_taken_from_later_lines(self):
        from plugins.platforms.email.adapter import _split_subject

        self.assertEqual(_split_subject("Intro\nSubject: x", "F"), ("F", "Intro\nSubject: x"))
        self.assertEqual(_split_subject("Subject: \nbody", "F"), ("F", "Subject: \nbody"))

    def test_subject_strips_control_chars_and_caps_length(self):
        from plugins.platforms.email.adapter import _split_subject

        self.assertEqual(_split_subject("Subject: A\x07B\r\nbody", "F"), ("AB", "body"))
        subject, body = _split_subject("Subject: " + "x" * 400 + "\nbody", "F")
        self.assertEqual(subject, "x" * 250)
        self.assertEqual(body, "body")

    def test_subject_falls_back_to_job_name(self):
        from plugins.platforms.email.adapter import _subject_from_thread_name

        self.assertEqual(_subject_from_thread_name("Hermes — Weekly report"), "Weekly report")
        self.assertEqual(_subject_from_thread_name(None), "Hermes Agent")
        self.assertEqual(_subject_from_thread_name(""), "Hermes Agent")
        self.assertEqual(_subject_from_thread_name("  Plain name  "), "Plain name")

    def test_standalone_send_uses_subject_rule(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import _standalone_send

        with patch("smtplib.SMTP") as mock_smtp, patch.dict(os.environ, ENV):
            result = asyncio.run(_standalone_send(PlatformConfig(enabled=True), "user@test.com", "Subject: Status\n\nAll good"))
            self.assertTrue(result["success"])
            msg = sent_message(mock_smtp)
            self.assertEqual(msg["Subject"], "Status")
            body = msg.get_payload(decode=True).decode("utf-8")
            self.assertEqual(body, "All good")
            self.assertNotIn("Subject:", body)

        with patch("smtplib.SMTP") as mock_smtp, patch.dict(os.environ, ENV):
            asyncio.run(_standalone_send(PlatformConfig(enabled=True), "user@test.com", "No subject line here"))
            self.assertEqual(sent_message(mock_smtp)["Subject"], "Hermes Agent")


def _html_of(msg):
    """Decoded text/html payload of *msg*, or None."""
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            return part.get_payload(decode=True).decode("utf-8")
    return None


class TestHtmlPart(unittest.TestCase):
    """MAIL-02: ``platforms.email.html`` adds a text/html alternative rendered from Markdown."""

    def setUp(self):
        import plugins.platforms.email.adapter as adapter_module

        self.adapter_module = adapter_module
        adapter_module._MD_RENDERER = None

    def tearDown(self):
        self.adapter_module._MD_RENDERER = None

    def _send(self, adapter, body):
        with patch("smtplib.SMTP") as mock_smtp:
            adapter._send_email("user@test.com", body, None)
            return sent_message(mock_smtp)

    def test_html_part_is_multipart_alternative_when_enabled(self):
        body = "# Title\n\n- one\n- two"
        msg = self._send(make_adapter(html=True), body)
        types = [p.get_content_type() for p in msg.walk()]
        self.assertEqual(types, ["multipart/mixed", "multipart/alternative", "text/plain", "text/html"])
        plain = [p for p in msg.walk() if p.get_content_type() == "text/plain"][0]
        self.assertEqual(plain.get_payload(decode=True).decode("utf-8"), body)
        html = _html_of(msg)
        self.assertIn("<h1>Title</h1>", html)
        self.assertIn("<li>one</li>", html)

    def test_html_escapes_raw_html(self):
        html = _html_of(self._send(make_adapter(html=True), "<script>alert(1)</script> and [x](javascript:alert(1))"))
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script", html)
        self.assertNotIn('href="javascript:', html)

    def test_html_image_rendered_as_link(self):
        html = _html_of(self._send(make_adapter(html=True), "![pixel](http://tracker.example.com/p.gif)"))
        self.assertNotIn("<img", html)
        self.assertIn('<a href="http://tracker.example.com/p.gif">pixel</a>', html)  # a stray "!" before it is fine

    def test_no_html_part_when_flag_off(self):
        msg = self._send(make_adapter(), "# Title\n\n- one")
        self.assertEqual([p.get_content_type() for p in msg.walk()], ["multipart/mixed", "text/plain"])

    def test_html_nested_in_mixed_with_attachment(self):
        import tempfile

        adapter = make_adapter(html=True)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"doc")
            tmp_path = f.name
        try:
            with patch("smtplib.SMTP") as mock_smtp:
                adapter._send_email_with_attachment("user@test.com", "**See** the file", tmp_path)
                msg = sent_message(mock_smtp)
        finally:
            os.unlink(tmp_path)
        self.assertEqual(msg.get_content_type(), "multipart/mixed")
        top = msg.get_payload()
        self.assertEqual([p.get_content_type() for p in top], ["multipart/alternative", "application/octet-stream"])
        self.assertEqual([p.get_content_type() for p in top[0].get_payload()], ["text/plain", "text/html"])
        self.assertIn("attachment", top[1]["Content-Disposition"])

    def test_html_flag_accepts_top_level_and_extra_spelling(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter

        with patch.dict(os.environ, ENV):
            top = EmailAdapter(PlatformConfig.from_dict({"enabled": True, "html": True}))
            nested = EmailAdapter(PlatformConfig.from_dict({"enabled": True, "extra": {"html": "true"}}))
            default = EmailAdapter(PlatformConfig.from_dict({"enabled": True}))
        self.assertIs(top._html, True)
        self.assertIs(nested._html, True)
        self.assertIs(default._html, False)

    def test_html_falls_back_to_plain_when_markdown_it_missing(self):
        import sys

        adapter = make_adapter(html=True)
        with patch.dict(sys.modules, {"markdown_it": None}), \
                self.assertLogs("plugins.platforms.email.adapter", "WARNING") as logs:
            first = self._send(adapter, "# Title")
            second = self._send(adapter, "# Again")
        for msg in (first, second):
            self.assertEqual([p.get_content_type() for p in msg.walk()], ["multipart/mixed", "text/plain"])
        warnings = [line for line in logs.output if "markdown-it-py" in line]
        self.assertEqual(len(warnings), 1)

    def test_standalone_send_html_when_enabled(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import _standalone_send

        with patch("smtplib.SMTP") as mock_smtp, patch.dict(os.environ, ENV):
            result = asyncio.run(_standalone_send(PlatformConfig(enabled=True, extra={"html": True}), "user@test.com",
                                                  "Subject: Status\n\n# All good"))
            self.assertTrue(result["success"])
            msg = sent_message(mock_smtp)
        self.assertEqual(msg["Subject"], "Status")
        self.assertEqual([p.get_content_type() for p in msg.walk()], ["multipart/alternative", "text/plain", "text/html"])
        self.assertIn("<h1>All good</h1>", _html_of(msg))


if __name__ == "__main__":
    unittest.main()
