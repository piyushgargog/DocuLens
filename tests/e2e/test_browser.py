"""Browser end-to-end checks for the frontend (opt-in: DOCULENS_E2E=1).

Written after three frontend regressions that unit tests could not see: the
"Try asking" list that appeared, vanished and reappeared, a footer heart that
stopped beating, and a landing page that popped in all at once under the OS
reduced-motion setting.
"""

import json
from pathlib import Path

import pytest

SAMPLE_PDF = Path(__file__).resolve().parents[2] / "sample_docs" / "sample.pdf"

TAILORED = [
    "Which planet is the largest?",
    "What makes Earth able to support life?",
    "How many moons does Saturn have?",
    "Which planets are terrestrial?",
]

REASONING = "The passage on page 3 names Jupiter as the largest planet."
ANSWER_TOKENS = ["Jupiter is the largest ", "planet in the Solar System ", "[sample.pdf, Page 3]."]


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def fake_llm_endpoints(page):
    """Answer the LLM-backed endpoints in the browser, so no key is needed."""
    page.route("**/api/suggestions", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps({"questions": TAILORED})))

    stream = (
        _sse("sources", [{"page": 3, "doc": "sample.pdf", "score": 0.61,
                          "text": "Jupiter is the largest planet in the Solar System."}])
        + _sse("reasoning", {"text": REASONING})
        + _sse("route", {"provider": "Test", "model": "fake-model"})
        + "".join(_sse("token", {"text": t}) for t in ANSWER_TOKENS)
        + _sse("done", {})
    )
    page.route("**/api/ask/stream", lambda route: route.fulfill(
        status=200, content_type="text/event-stream", body=stream))


# Records every distinct state of the "Try asking" list as it mutates, so a
# list that shows buttons, drops them and shows different ones is caught even
# if the swap happens between two polls.
RECORD_SUGGESTION_STATES = """
window.__suggestionStates = [];
new MutationObserver(() => {
  const list = document.querySelector('.suggestion-list');
  if (!list) return;
  const state = {
    skeletons: list.querySelectorAll('.suggestion-skeleton').length,
    buttons: [...list.querySelectorAll('button.suggestion')].map(b => b.textContent),
  };
  const last = window.__suggestionStates[window.__suggestionStates.length - 1];
  if (!last || JSON.stringify(last) !== JSON.stringify(state)) window.__suggestionStates.push(state);
}).observe(document, { childList: true, subtree: true });
"""


@pytest.mark.parametrize("motion", ["no-preference", "reduce"])
def test_landing_renders_with_entrance_and_heartbeat(server, make_page, app_version, motion):
    page, errors = make_page(reduced_motion=motion)
    page.goto(server)
    title = page.locator("#upload-view .intro-title")
    title.wait_for(state="visible")

    # The landing page arrives with its settle-in animation in both modes
    # (v3.6.2: reduced motion must not flatten it to an instant pop).
    assert page.evaluate("t => getComputedStyle(t).animationName", title.element_handle()) == "settle"
    page.wait_for_function(
        "() => getComputedStyle(document.querySelector('#upload-view .intro-title')).opacity === '1'")

    # The footer heart beats (a scale animation), also under reduced motion.
    heart = page.locator(".heart")
    assert page.evaluate("h => getComputedStyle(h).animationName", heart.element_handle()) == "beat"
    assert page.evaluate("h => getComputedStyle(h).animationIterationCount", heart.element_handle()) == "infinite"

    assert page.locator(".version-badge").inner_text().strip() == f"v{app_version}"
    assert errors == []


def test_upload_shows_starters_once_and_streams_an_answer(server, make_page):
    page, errors = make_page()
    page.add_init_script(RECORD_SUGGESTION_STATES)
    fake_llm_endpoints(page)
    page.goto(server)

    page.set_input_files("#file-input", str(SAMPLE_PDF))
    page.locator("button.suggestion").first.wait_for(timeout=120_000)
    page.wait_for_function(f"() => document.querySelectorAll('button.suggestion').length === {len(TAILORED)}")

    states = page.evaluate("() => window.__suggestionStates")
    with_buttons = [i for i, s in enumerate(states) if s["buttons"]]
    assert with_buttons, states
    first = with_buttons[0]
    # Placeholders hold the spot before any question appears ...
    assert any(s["skeletons"] > 0 and not s["buttons"] for s in states[:first]), states
    # ... and the first questions shown are the final ones: nothing is shown,
    # removed and replaced (the v3.5.0-v3.6.0 double flash).
    assert all(s["buttons"] == TAILORED for s in states[first:]), states

    page.fill("#question-input", "Which planet is the largest?")
    page.keyboard.press("Enter")

    answer = page.locator(".answer").last
    answer.locator(".cite").first.wait_for()
    page.wait_for_function("() => !document.body.classList.contains('busy')")

    # The model's thinking is shown (display-only) and folded once the answer starts.
    thinking = answer.locator(".thinking")
    assert thinking.is_visible()
    assert REASONING in answer.locator(".thinking-text").text_content()
    assert "Jupiter is the largest planet" in answer.inner_text()
    assert answer.locator(".cite").count() >= 1

    assert errors == []


# ---------- v4: sign-in, saved documents, summaries, sections, warnings ----------

GUEST_ME = {
    "auth_enabled": True,
    "firebase": {"apiKey": "fake", "authDomain": "x.firebaseapp.com", "projectId": "x-project"},
    "user": None,
    "limits": {"tier": "guest", "max_docs": 1, "questions_per_day": 5},
}
USER_ME = {**GUEST_ME, "user": {"name": "Ada Lovelace", "email": "ada@example.com"}, "limits": {"tier": "user", "max_docs": 5, "questions_per_day": 200}}


def fake_me(page, payload):
    page.route("**/api/me", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(payload)))


def test_guest_sees_limits_and_a_sign_in_button(server, make_page):
    page, errors = make_page()
    fake_me(page, GUEST_ME)
    page.goto(server)
    page.locator("#signin-link:not([hidden])").wait_for()
    note = page.locator("#guest-note")
    note.wait_for(state="visible")
    assert "1 document" in note.inner_text() and "5 questions a day" in note.inner_text()
    assert page.locator("#user-chip").is_hidden()
    assert "not saved" in page.locator("#shelf-note").inner_text().lower()
    assert errors == []


def test_signed_in_user_sees_a_chip_and_saved_note(server, make_page):
    page, errors = make_page()
    fake_me(page, USER_ME)
    page.route("**/api/chats", lambda r: r.fulfill(status=200, content_type="application/json", body='{"chats": []}'))
    page.goto(server)
    page.locator("#user-chip:not([hidden])").wait_for()
    assert page.locator("#user-name").inner_text() == "Ada Lovelace"
    assert page.locator("#user-avatar").inner_text() == "A"
    assert page.locator("#signin-link").is_hidden() and page.locator("#guest-note").is_hidden()
    assert "saved to your account" in page.locator("#shelf-note").inner_text().lower()
    assert errors == []


def test_sign_in_is_hidden_when_it_is_not_configured(server, make_page):
    page, errors = make_page()
    page.goto(server)
    page.wait_for_load_state("networkidle")
    assert page.locator("#signin-link").is_hidden() and page.locator("#user-chip").is_hidden() and page.locator("#shelf-note").is_hidden()
    assert errors == []


def test_a_partial_summary_says_which_parts_were_not_read(server, make_page):
    page, errors = make_page()
    fake_llm_endpoints(page)
    page.route("**/api/summary", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({
        "filename": "sample.pdf",
        "summary": "- The document covers the Solar System [Pages 1-2].",
        "sources": [{"page": 1, "text": "The Solar System consists of the Sun.", "score": None, "section": "1 Overview"}],
        "coverage": {"complete": False, "skipped_ranges": ["Pages 3-4"], "failed_ranges": []},
        "answered_by": {"provider": "Test", "model": "fake-model"},
    })))
    page.goto(server)
    page.set_input_files("#file-input", str(SAMPLE_PDF))
    page.locator("#doc-list .doc-item").first.wait_for(timeout=120_000)
    page.locator(".doc-summary").first.click()
    page.locator(".msg-note", has_text="partial").wait_for()
    note = page.locator(".msg-note", has_text="partial").inner_text()
    assert "Pages 3-4" in note and "not read" in note
    assert errors == []


def test_sources_show_their_section_when_there_is_one(server, make_page):
    page, errors = make_page()
    fake_llm_endpoints(page)
    page.route("**/api/ask/stream", lambda route: route.fulfill(status=200, content_type="text/event-stream", body=(
        _sse("sources", [{"page": 2, "score": 0.5, "section": "3.2 Attention", "text": "Attention maps a query and a set of key-value pairs to an output."}])
        + _sse("token", {"text": "It maps queries to outputs [Page 2]."}) + _sse("done", {}))))
    page.goto(server)
    page.set_input_files("#file-input", str(SAMPLE_PDF))
    page.locator("button.suggestion").first.wait_for(timeout=120_000)
    page.fill("#question-input", "What is attention?")
    page.keyboard.press("Enter")
    page.locator(".slip-section:not([hidden])").first.wait_for(state="attached")
    assert page.locator(".slip-section:not([hidden])").first.text_content() == "3.2 Attention"
    assert errors == []


def test_ingest_warnings_are_shown_as_notes(server, make_page):
    page, errors = make_page()
    doc = {"id": "d1", "filename": "scan.pdf", "num_pages": 6, "num_chunks": 3}
    # (route.fetch() cannot replay a binary multipart upload, so the response is faked whole)
    page.route("**/api/ingest", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(
        {**doc, "documents": [doc], "warnings": ["Pages 4-6 could not be read (blank, too faint or too noisy).", "Page 2 was hard to read; the text may contain errors."]})))
    fake_llm_endpoints(page)
    page.goto(server)
    page.set_input_files("#file-input", str(SAMPLE_PDF))
    page.locator(".msg-note", has_text="Pages 4-6 could not be read").wait_for(timeout=30_000)
    assert page.locator(".msg-note", has_text="hard to read").count() == 1
    assert errors == []


def test_documents_the_server_could_not_restore_are_refreshed_away(server, make_page):
    page, errors = make_page()
    fake_llm_endpoints(page)
    page.goto(server)
    page.set_input_files("#file-input", str(SAMPLE_PDF))
    page.locator("button.suggestion").first.wait_for(timeout=120_000)
    page.route("**/api/ask/stream", lambda route: route.fulfill(status=409, content_type="application/json", body=json.dumps(
        {"error": "These saved documents could no longer be restored and were removed: a.pdf. Please upload them again."})))
    page.route("**/api/session", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"documents": [], "history": []})))
    page.fill("#question-input", "anything?")
    page.keyboard.press("Enter")
    page.locator("#upload-view").wait_for(state="visible")
    assert page.locator("#doc-list .doc-item").count() == 0
    assert all("409" in e or "Failed to load resource" in e for e in errors)  # the browser logs the 409 itself; nothing else


# ---------- v4: log-in dialog and saved chats ----------

CHATS = [
    {"id": "chatAAAAAAAA", "title": "How many attention heads?", "created": 1, "updated": 3},
    {"id": "chatBBBBBBBB", "title": "Dropout settings", "created": 1, "updated": 2},
]
TURNS = {"chatAAAAAAAA": [{"question": "How many heads?", "answer": "Eight heads [Page 5]."}], "chatBBBBBBBB": [{"question": "Dropout?", "answer": "0.1 [Page 8]."}]}
DOCS = [{"id": "d1", "filename": "paper.pdf", "num_pages": 15, "num_chunks": 66}]


def fake_chats_api(page):
    ok = lambda body: dict(status=200, content_type="application/json", body=json.dumps(body))  # noqa: E731
    state = {"chats": list(CHATS), "deleted": [], "renamed": []}
    page.route("**/api/session", lambda r: r.fulfill(**ok({"documents": DOCS, "history": []})))
    page.route("**/api/suggestions", lambda r: r.fulfill(**ok({"questions": []})))
    page.route("**/api/chats", lambda r: r.fulfill(**ok({"chats": state["chats"]})))
    page.route("**/api/chat?*", lambda r: r.fulfill(**ok({"chat": CHATS[0], "history": TURNS[r.request.url.split("id=")[1]]})))

    def rename(route):
        body = json.loads(route.request.post_data)
        state["renamed"].append(body)
        route.fulfill(**ok({"chat": {**CHATS[0], "title": body["title"]}}))

    def delete(route):
        state["deleted"].append(json.loads(route.request.post_data)["id"])
        route.fulfill(**ok({"ok": True}))

    page.route("**/api/chats/rename", rename)
    page.route("**/api/chats/delete", delete)
    return state


def test_login_opens_a_dialog_with_the_google_button(server, make_page):
    page, errors = make_page()
    fake_me(page, GUEST_ME)
    page.goto(server)
    page.locator("#signin-link:not([hidden])").wait_for()
    page.click("#signin-link")
    dialog = page.locator("#signin-dialog")
    dialog.wait_for(state="visible")
    assert "Continue with Google" in page.locator("#google-btn").inner_text() and "Log in to DocuLens" in dialog.inner_text()
    page.get_by_role("button", name="Not now").click()
    dialog.wait_for(state="hidden")
    assert errors == []


def test_signed_in_users_get_a_chat_sidebar_with_their_latest_chat_open(server, make_page):
    page, errors = make_page()
    fake_me(page, USER_ME)
    state = fake_chats_api(page)
    page.goto(server)
    page.locator(".chat-item").first.wait_for()
    assert page.locator(".chat-item").count() == 2
    assert page.locator(".chat-item.active .chat-open").inner_text() == "How many attention heads?"
    page.locator(".msg", has_text="Eight heads").first.wait_for()
    page.locator(".chat-item", has_text="Dropout settings").locator(".chat-open").click()
    page.locator(".msg", has_text="0.1").first.wait_for()
    assert page.locator(".msg", has_text="Eight heads").count() == 0  # the other chat's turns are gone
    page.click("#new-chat-btn")
    page.locator(".msg-note", has_text="New chat").wait_for()
    assert page.locator(".chat-item.active").count() == 0
    assert errors == [] and state["deleted"] == []


def test_a_chat_can_be_renamed_and_deleted_with_a_confirmation_click(server, make_page):
    page, errors = make_page()
    fake_me(page, USER_ME)
    state = fake_chats_api(page)
    page.goto(server)
    page.locator(".chat-item").first.wait_for()
    row = page.locator(".chat-item", has_text="Dropout settings")
    row.hover()
    row.get_by_role("button", name="Rename chat").click()
    box = page.locator(".chat-rename")
    box.fill("Regularisation")
    box.press("Enter")
    page.wait_for_function("() => true")
    assert state["renamed"] == [{"id": "chatBBBBBBBB", "title": "Regularisation"}]
    row = page.locator(".chat-item", has_text="Regularisation").or_(page.locator(".chat-item", has_text="Dropout settings"))
    row.first.hover()
    row.first.get_by_role("button", name="Delete chat").click()
    assert state["deleted"] == []  # the first click only asks
    row.first.locator(".chat-act.danger").click()
    page.wait_for_function("() => document.querySelectorAll('.chat-item').length === 1")
    assert state["deleted"] == ["chatBBBBBBBB"] and errors == []
