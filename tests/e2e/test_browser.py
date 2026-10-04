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
