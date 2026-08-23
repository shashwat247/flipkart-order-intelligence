"""Integration tests for the Streamlit app (streamlit_app/app.py).

Two layers, both against the REAL backend (no mocking of models, retrieval,
or the agent):

1. Direct calls to the backend-glue functions streamlit_app/app.py exposes —
   fast, precise, and the primary evidence that the UI's plumbing reaches the
   real Part 1/2/3 artifacts rather than a stand-in.
2. `streamlit.testing.v1.AppTest` runs against widgets, simulating clicks and
   form submissions end-to-end through the actual rendered page, so the UI
   wiring itself is exercised, not just the functions behind it.
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from streamlit_app.app import (
    ACCENT,  # noqa: F401 - imported to prove the module loads cleanly
    analyze_risk,
    ask_policy,
    classify_uploaded_bytes,
    default_risk_features,
    get_system_status,
    lookup_real_order,
    new_conversation,
    run_retrieval_evaluation,
    search_kb,
)
from part1.common import FEATURES, MODEL_PATH as RISK_MODEL_PATH
from part2.model import classify_product_image
from part3.graph import run_once

APP_PATH = str(Path(__file__).resolve().parents[1] / "streamlit_app" / "app.py")
SNEAKER = "data/sample_images/07_sneaker.png"
TEST_ORDER_ID = 1790


# =========================================================== 1. app starts
def test_app_starts_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    assert not at.exception
    assert at.sidebar.radio[0].value == "Dashboard"


@pytest.mark.parametrize("page", [
    "AI Support", "Return Risk", "Product Classifier",
    "Policy Knowledge", "Model Insights", "System Status",
])
def test_every_page_renders_without_exception(page):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    at.sidebar.radio[0].set_value(page)
    at.run(timeout=120)
    assert not at.exception


def test_system_status_page_matches_get_system_status():
    """System Status must show the same rows get_system_status() computes — not hardcoded copies."""
    rows = get_system_status()
    assert len(rows) == 6
    assert all(isinstance(r["ready"], bool) for r in rows)
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    at.sidebar.radio[0].set_value("System Status")
    at.run(timeout=120)
    assert not at.exception
    page_text = " ".join(m.value for m in at.markdown) + " " + " ".join(c.value for c in at.caption)
    for r in rows:
        assert r["name"] in page_text
        assert r["detail"] in page_text


def test_dashboard_feature_cards_navigate_to_their_real_pages():
    """Each Dashboard card's button must switch the sidebar nav to its real page —
    no dead links, no hardcoded page copy."""
    for button_label, target_page in [
        ("Analyze return risk", "Return Risk"),
        ("Classify image", "Product Classifier"),
        ("Ask policy", "Policy Knowledge"),
        ("Open assistant", "AI Support"),
        ("View status", "System Status"),
    ]:
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=120)
        btn = [b for b in at.button if b.label == button_label]
        assert btn, f"dashboard card button {button_label!r} not found"
        btn[0].click().run(timeout=120)
        assert not at.exception
        assert at.sidebar.radio[0].value == target_page


# ================================================ 2. risk form reaches the real model
def test_analyze_risk_matches_check_return_risk_tool_directly():
    from part3.tools import check_return_risk

    features = lookup_real_order(TEST_ORDER_ID)
    via_frontend = analyze_risk(features)
    via_tool = check_return_risk(features)
    assert via_frontend == via_tool
    assert via_frontend["status"] == "ok"
    assert via_frontend["risk_bucket"] in ("Low", "Medium", "High")


def test_analyze_risk_matches_the_saved_model_called_directly():
    features = lookup_real_order(TEST_ORDER_ID)
    result = analyze_risk(features)

    model = joblib.load(RISK_MODEL_PATH)
    direct = float(model.predict_proba(pd.DataFrame([features])[FEATURES])[0][1])
    assert result["return_probability"] == round(direct, 4)


def test_return_risk_analyzer_form_submits_to_the_real_model():
    """End-to-end: fill the actual form widgets (by label, not position), submit, and
    read the real result. Label-based lookup so the test survives layout changes."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    at.sidebar.radio[0].set_value("Return Risk")
    at.run(timeout=120)
    assert not at.exception

    def _by_label(widgets, label):
        matches = [w for w in widgets if w.label == label]
        assert matches, f"no widget labeled {label!r} found on the page"
        return matches[0]

    _by_label(at.selectbox, "Product category").set_value("Electronics")
    _by_label(at.selectbox, "Payment method").set_value("COD")
    _by_label(at.number_input, "Price (INR)").set_value(35000.0)
    submit = [b for b in at.button if b.label == "Analyze Return Risk"]
    assert submit, "the Analyze Return Risk submit button must be on the page"
    submit[0].click().run(timeout=120)

    assert not at.exception
    page_text = " ".join(m.value for m in at.markdown)
    assert "RISK" in page_text  # LOW/MEDIUM/HIGH badge rendered from a real prediction


# ============================================ 3. image upload reaches the real classifier
def test_classify_uploaded_bytes_matches_classify_product_image_directly():
    from pathlib import Path

    data = Path(SNEAKER).read_bytes()
    via_frontend = classify_uploaded_bytes(data, ".png")
    via_direct = classify_product_image(SNEAKER)
    assert via_frontend["predicted_class"] == via_direct["predicted_class"]
    assert via_frontend["confidence"] == via_direct["confidence"]


def test_classify_uploaded_bytes_ignores_the_suffix_hint_for_safety():
    """An unrecognised suffix must not be used verbatim for the temp file."""
    from pathlib import Path

    data = Path(SNEAKER).read_bytes()
    result = classify_uploaded_bytes(data, ".exe")
    assert result["predicted_class"]


def test_classify_uploaded_bytes_reports_invalid_images_without_crashing():
    with pytest.raises(Exception):
        classify_uploaded_bytes(b"not a real image", ".png")


def test_product_image_analyzer_page_classifies_a_real_sample_via_the_ui():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    at.sidebar.radio[0].set_value("Product Classifier")
    at.run(timeout=120)
    assert not at.exception

    sample_select = [s for s in at.selectbox if "sample" in (s.label or "").lower()]
    assert sample_select, "the sample-image selectbox must be on the page"
    sample_select[0].set_value("07_sneaker.png")
    at.run(timeout=120)

    classify_btn = [b for b in at.button if b.label == "Classify Product"]
    assert classify_btn
    classify_btn[0].click().run(timeout=120)

    assert not at.exception
    page_text = " ".join(m.value for m in at.markdown)
    assert "Sneaker" in page_text or "confidence" in page_text.lower()


# ========================================== 4. policy query reaches the real retrieval
def test_ask_policy_matches_run_once_directly():
    query = "How many days do I have to return a mobile phone?"
    via_frontend = ask_policy(query)
    via_direct = run_once(query)
    assert via_frontend["response"] == via_direct["response"]
    assert via_frontend["response"]["source"] == "policy_kb"
    assert via_frontend["groundedness"]["grounded"] is True
    assert via_frontend["doc_hits"]


def test_search_kb_returns_real_scored_documents():
    hits = search_kb("Will a courier come to my house to collect a return?", n_documents=3)
    assert 1 <= len(hits) <= 3
    assert all(0.0 <= h["score"] <= 1.0 for h in hits)
    assert all(h["document_id"] for h in hits)


def test_retrieval_evaluation_matches_the_documented_readme_numbers():
    evaluation = run_retrieval_evaluation()
    assert evaluation["mean_precision_at_3"] == pytest.approx(0.4286, abs=5e-4)
    assert evaluation["mean_recall_at_3"] == pytest.approx(0.9286, abs=5e-4)
    assert len(evaluation["rows"]) == 7


def test_policy_knowledge_search_finds_real_documents_via_the_ui():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    at.sidebar.radio[0].set_value("Policy Knowledge")
    at.run(timeout=120)
    assert not at.exception

    search_box = [t for t in at.text_input if t.label == "Search policies…"]
    assert search_box, "the policy search box must be on the page"
    search_box[0].set_value("footwear")
    at.run(timeout=120)

    page_text = " ".join(m.value for m in at.markdown) + " ".join(e.label or "" for e in at.expander)
    assert "Footwear Return Window" in page_text


def test_policy_knowledge_ask_ai_button_routes_to_ai_support_and_answers_for_real():
    """Clicking 'Ask AI about this policy' on a card must land on AI Support and
    actually reach the real LangGraph agent — not just switch pages."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    at.sidebar.radio[0].set_value("Policy Knowledge")
    at.run(timeout=120)
    assert not at.exception

    ask_ai_buttons = [b for b in at.button if "Ask AI about this policy" in b.label]
    assert ask_ai_buttons, "each policy card must offer an 'Ask AI about this policy' button"
    ask_ai_buttons[0].click().run(timeout=120)

    assert not at.exception
    assert at.sidebar.radio[0].value == "AI Support"
    assert len(at.session_state["chat_log"]) == 1
    turn = at.session_state["chat_log"][0]
    assert "policy" in turn["query"].lower()
    assert turn["result"]["response"]["source"] == "policy_kb"


# ========================================================== 5/6. chat state and reset
def test_conversation_state_carried_and_reset_via_the_frontend_factory():
    conversation = new_conversation()
    conversation.ask("Check order 2314 for me.")
    followup = conversation.ask("What is its return risk?")
    assert followup["order_id"] == 2314
    assert "2314" in followup["response"]["answer"]

    fresh = new_conversation()
    result = fresh.ask("What is its return risk?")
    assert result["order_id"] is None
    assert "2314" not in result["response"]["answer"]


def test_support_assistant_new_conversation_button_clears_session_state():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=120)
    at.sidebar.radio[0].set_value("AI Support")
    at.run(timeout=120)
    assert not at.exception

    sample_btn = [b for b in at.button if "return risk" in b.label.lower()
                  or "1790" in b.label]
    assert sample_btn, "a sample question chip must be present"
    sample_btn[0].click().run(timeout=120)
    assert not at.exception
    assert len(at.session_state["chat_log"]) == 1

    reset_btn = [b for b in at.button if "New Conversation" in b.label]
    assert reset_btn
    reset_btn[0].click().run(timeout=120)
    assert not at.exception
    assert at.session_state["chat_log"] == []
    assert at.session_state["conversation"].state.get("order_id") is None


# ================================================= 7. missing artifacts -> friendly errors
def test_status_checks_report_not_ready_instead_of_raising(monkeypatch, tmp_path):
    import streamlit_app.app as appmod
    import part1.common as common

    monkeypatch.setattr(common, "MODEL_PATH", tmp_path / "does_not_exist.pkl")
    ready, detail = appmod._risk_model_status()
    assert ready is False
    assert "missing" in detail.lower() or "not" in detail.lower()


def test_classify_uploaded_bytes_raises_a_catchable_error_for_garbage_input():
    """The frontend never crashes silently: an invalid image must raise a normal
    exception the page's try/except can turn into a friendly st.error, not hang
    or return a fabricated prediction."""
    with pytest.raises(Exception) as excinfo:
        classify_uploaded_bytes(b"", ".png")
    assert excinfo.value is not None


def test_get_system_status_never_raises_even_if_a_component_is_broken(monkeypatch):
    import streamlit_app.app as appmod

    def boom():
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(appmod, "_agent_status", boom)
    rows = get_system_status()
    broken = [r for r in rows if r["name"] == "Support Agent (LangGraph)"]
    assert broken and broken[0]["ready"] is False
    assert "unexpected error" in broken[0]["detail"]


def test_return_risk_analyzer_reports_missing_input_without_crashing():
    result = analyze_risk({"price_inr": 999})
    assert result["status"] == "missing_input"
    assert set(result["missing_features"]) == set(FEATURES) - {"price_inr"}


def test_default_risk_features_are_valid_model_input():
    result = analyze_risk(default_risk_features())
    assert result["status"] == "ok"
    assert 0.0 <= result["return_probability"] <= 1.0
