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


def _store_path() -> Path:
    from plugins.plugin_storage import plugin_data_dir

    return plugin_data_dir("email") / "threads.json"


class TestThreadStore(unittest.TestCase):
    """The persistent thread store and the one root sanitiser both ends share."""

    def test_root_sanitised_for_session_key(self):
        from gateway.session import _is_path_unsafe
        from plugins.platforms.email.adapter import _root_key

        self.assertEqual(_root_key("<hermes-abc@test.com>"), "hermes-abc@test.com")
        self.assertEqual(_root_key("  hermes-abc@test.com "), "hermes-abc@test.com")
        hashed = ["<a/../b@x.com>", "<a b@x.com>", "<" + "x" * 200 + "@x.com>", "<\\evil@x.com>", "<c:x@y>"]
        for raw in hashed:
            key = _root_key(raw)
            self.assertTrue(key.startswith("h-"), raw)
            self.assertEqual(len(key), 26, raw)
            self.assertEqual(_root_key(raw), key, raw)
        self.assertEqual(len({_root_key(raw) for raw in hashed}), len(hashed))
        for raw in hashed + ["<hermes-abc@test.com>"]:
            self.assertFalse(_is_path_unsafe(_root_key(raw), strict=False), raw)

    def test_message_ids_parses_folded_header(self):
        from plugins.platforms.email.adapter import _message_ids

        self.assertEqual(_message_ids("<a@x>\r\n\t<b@y> junk <c@z>"), ["<a@x>", "<b@y>", "<c@z>"])
        self.assertEqual(_message_ids(None), [])
        self.assertEqual(_message_ids(""), [])

    def test_thread_store_survives_new_adapter_instance(self):
        first = make_adapter()
        first._store.add_pending_root("k1", "<k1>", "Hermes — Report", "user@test.com")
        second = make_adapter()
        root = second._store.get_root("k1")
        self.assertEqual(root["name"], "Hermes — Report")
        self.assertIs(root["pending"], True)
        self.assertTrue(second._store.is_pending("k1"))

    def test_corrupt_store_resets_to_empty(self):
        path = _store_path()
        path.write_text("{not json", encoding="utf-8")
        with self.assertLogs("plugins.platforms.email.adapter", "WARNING") as logs:
            adapter = make_adapter()
        self.assertTrue(any("threads.json" in line for line in logs.output))
        self.assertIsNone(adapter._store.get_root("k1"))
        adapter._store.add_pending_root("k1", "<k1>", "Hermes — Report", "user@test.com")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("k1", data["roots"])

    def test_store_prunes_old_roots(self):
        import time

        now = time.time()
        root = {"header": "", "name": "n", "subject": "", "job_id": "", "chat_id": "user@test.com", "pending": False}
        _store_path().write_text(json.dumps({
            "version": 1,
            "roots": {"old": dict(root, header="<old>", created=now - 91 * 86400),
                      "new": dict(root, header="<new>", created=now - 10 * 86400)},
            "index": {"old": "old", "m-old@x": "old", "new": "new", "m-new@x": "new"},
            "mailboxes": {},
        }), encoding="utf-8")
        store = make_adapter()._store
        self.assertIsNone(store.get_root("old"))
        self.assertIsNone(store.root_for(["<m-old@x>"]))
        self.assertEqual(store.get_root("new")["header"], "<new>")
        self.assertEqual(store.root_for(["<m-new@x>"]), "new")
        store.index_message("<m2@x>", "new")  # the next write persists the pruned state
        on_disk = json.loads(_store_path().read_text(encoding="utf-8"))
        self.assertEqual(set(on_disk["roots"]), {"new"})
        self.assertNotIn("m-old@x", on_disk["index"])
        self.assertEqual(on_disk["index"]["m2@x"], "new")

    def test_store_index_resolves_first_known_id(self):
        store = make_adapter()._store
        store.add_pending_root("k1", "<k1>", "Hermes — Report", "user@test.com")
        store.index_message("<m2@x>", "k1")
        self.assertEqual(store.root_for(["<unknown@x>", "<m2@x>"]), "k1")
        self.assertEqual(store.root_for(["<k1>"]), "k1")
        self.assertIsNone(store.root_for([]))
        self.assertIsNone(store.root_for(["<unknown@x>"]))

    def test_store_concurrent_writes_keep_valid_json(self):
        import threading

        store = make_adapter()._store
        store.add_pending_root("k1", "<k1>", "Hermes — Report", "user@test.com")

        def worker(n):
            for i in range(50):
                store.index_message(f"<t{n}-{i}@x>", "k1")

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        data = json.loads(_store_path().read_text(encoding="utf-8"))
        ids = {f"t{n}-{i}@x" for n in range(4) for i in range(50)}
        self.assertEqual(ids - set(data["index"]), set())
        self.assertEqual(len(ids), 200)


class TestOutboundThreads(unittest.TestCase):
    """MAIL-01 end to end and the outbound half of MAIL-03: minted roots, fresh first mail, References chain."""

    def _mint(self, adapter, name="Hermes — Report"):
        return asyncio.run(adapter.create_handoff_thread("user@test.com", name))

    def _send(self, adapter, content, **kwargs):
        with patch("smtplib.SMTP") as mock_smtp:
            result = asyncio.run(adapter.send("user@test.com", content, **kwargs))
            msg = sent_message(mock_smtp) if mock_smtp.return_value.send_message.called else None
        return result, msg

    @staticmethod
    def _plain(msg):
        return [p for p in msg.walk() if p.get_content_type() == "text/plain"][0].get_payload(decode=True).decode("utf-8")

    def test_create_handoff_thread_mints_pending_root(self):
        import re

        adapter = make_adapter()
        key = asyncio.run(adapter.create_handoff_thread("User@Test.com", "Hermes — Report"))
        self.assertRegex(key, r"^hermes-[0-9a-f]{12}@test\.com$")
        root = adapter._store.get_root(key)
        self.assertIs(root["pending"], True)
        self.assertEqual(root["name"], "Hermes — Report")
        self.assertEqual(root["chat_id"], "user@test.com")
        self.assertEqual(root["header"], "<" + key + ">")
        self.assertIsNotNone(re.fullmatch(r"[^<>\s/\\]+", key))

    def test_pending_root_sent_as_fresh_mail_with_minted_message_id(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        result, msg = self._send(adapter, "Subject: Report — Nov 9\n\n- item", metadata={"thread_id": key, "job_id": "j1"})
        self.assertTrue(result.success)
        self.assertEqual(msg["Message-ID"], "<" + key + ">")
        self.assertEqual(msg["Subject"], "Report — Nov 9")
        self.assertIsNone(msg["In-Reply-To"])
        self.assertIsNone(msg["References"])
        self.assertNotIn("Subject:", self._plain(msg))
        self.assertEqual(self._plain(msg), "- item")
        self.assertEqual(result.message_id, "<" + key + ">")
        root = adapter._store.get_root(key)
        self.assertIs(root["pending"], False)
        self.assertEqual(root["job_id"], "j1")
        self.assertEqual(root["subject"], "Report — Nov 9")

    def test_pending_root_subject_falls_back_to_job_name(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        _, msg = self._send(adapter, "- item", metadata={"thread_id": key, "job_id": "j1"})
        self.assertEqual(msg["Subject"], "Report")
        self.assertEqual(msg["Message-ID"], "<" + key + ">")

    def test_pending_root_ignores_per_sender_context(self):
        adapter = make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Old", "message_id": "<old@test.com>"}
        key = self._mint(adapter)
        _, msg = self._send(adapter, "- item", metadata={"thread_id": key, "job_id": "j1"})
        self.assertIsNone(msg["In-Reply-To"])
        self.assertIsNone(msg["References"])
        self.assertNotIn("Old", msg["Subject"])
        self.assertEqual(msg["Subject"], "Report")

    def test_second_chunk_threads_under_root(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        meta = {"thread_id": key, "job_id": "j1"}
        self._send(adapter, "Subject: Report — Nov 9\n\n- part one", metadata=meta)
        result, msg = self._send(adapter, "- part two", metadata=meta)
        self.assertEqual(msg["Subject"], "Re: Report — Nov 9")
        self.assertEqual(msg["In-Reply-To"], "<" + key + ">")
        self.assertEqual(msg["References"], "<" + key + ">")
        self.assertNotEqual(msg["Message-ID"], "<" + key + ">")
        self.assertEqual(result.message_id, msg["Message-ID"])
        self.assertEqual(adapter._store.root_for([msg["Message-ID"]]), key)
        self.assertEqual(adapter._store.get_root(key)["job_id"], "j1")

    def test_reply_sends_full_references_chain(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        self._send(adapter, "Subject: Report — Nov 9\n\n- item", metadata={"thread_id": key, "job_id": "j1"})
        root_id = "<" + key + ">"
        adapter._root_context[("user@test.com", key)] = {
            "subject": "Report — Nov 9", "message_id": "<r2@test.com>", "references": [root_id, "<h1@test.com>"]}
        _, msg = self._send(adapter, "answer", reply_to="<r2@test.com>", metadata={"thread_id": key})
        self.assertEqual(msg["In-Reply-To"], "<r2@test.com>")
        self.assertEqual(msg["References"], f"{root_id} <h1@test.com> <r2@test.com>")
        self.assertEqual(msg["Subject"], "Re: Report — Nov 9")

    def test_references_chain_capped(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        self._send(adapter, "- item", metadata={"thread_id": key, "job_id": "j1"})
        root_id = "<" + key + ">"
        refs = [root_id] + [f"<m{i}@test.com>" for i in range(1, 40)]
        self.assertEqual(len(refs), 40)
        adapter._root_context[("user@test.com", key)] = {"subject": "Report", "message_id": "<m39@test.com>",
                                                         "references": refs}
        _, msg = self._send(adapter, "answer", reply_to="<p@test.com>", metadata={"thread_id": key})
        ids = msg["References"].split()
        self.assertEqual(len(ids), 20)
        self.assertEqual(ids[0], root_id)
        self.assertEqual(ids[1:], (refs + ["<p@test.com>"])[-19:])
        self.assertEqual(ids[-1], "<p@test.com>")
        self.assertEqual(msg["In-Reply-To"], "<p@test.com>")

    def test_send_without_thread_unchanged(self):
        adapter = make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Project question", "message_id": "<original@test.com>"}
        result, msg = self._send(adapter, "x")
        self.assertTrue(result.success)
        self.assertEqual(msg["Subject"], "Re: Project question")
        self.assertEqual(msg["In-Reply-To"], "<original@test.com>")
        self.assertEqual(msg["References"], "<original@test.com>")
        self.assertEqual(adapter._store.root_for([msg["Message-ID"]]), None)

    def test_failed_smtp_leaves_root_pending(self):
        adapter = make_adapter()
        key = self._mint(adapter)
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value.send_message.side_effect = OSError("smtp down")
            result = asyncio.run(adapter.send("user@test.com", "Subject: Report\n\nbody",
                                              metadata={"thread_id": key, "job_id": "j1"}))
        self.assertIs(result.success, False)
        root = adapter._store.get_root(key)
        self.assertIs(root["pending"], True)
        self.assertEqual(root["job_id"], "")
        self.assertIsNone(adapter._store.root_for(["<unrelated@test.com>"]))


if __name__ == "__main__":
    unittest.main()
