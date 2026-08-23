"""Tests proving the Part 3 agent is not a predefined-question chatbot.

Every query below is deliberately absent from `part3/prompts.py`'s few-shot
exemplars and from the original demo/transcript questions -- typos, informal
phrasing, greetings, multi-clause questions, product-catalog questions and
cross-turn slot-filling. None of these are answered by matching against a
list of known questions: each one is routed by the same embedding-based
intent classifier and RAG pipeline the original three required intents use
(see part3/graph.py).
"""

import json

import pytest

from part3.config import VALID_SOURCES
from part3.graph import Conversation, run_once

REFUSAL_PHRASES = ("against policy", "against our policy")


def _answer(query: str, image: str | None = None) -> dict:
    return run_once(query, image_path=image)["response"]


# ---------------------------------------------------- never a blanket refusal
def test_never_falls_back_to_a_literal_against_policy_refusal():
    """The exact failure mode this upgrade targets: a demo-question whitelist
    that answers anything unrecognised with a canned "against policy" line.
    """
    unseen_questions = [
        "Hey, what can you help me with?",
        "Could you explain the footwear return rules in simple language?",
        "What happens if I open a pair of headphones and then decide I don't want them?",
        "My package arrived late and I paid cash on delivery. What should I know?",
        "Can I return something after a week?",
        "How are footwear and electronics returns different?",
        "i ordered shoes and they're damaged, what now?",
        "can i retun shoes after 8 day",
        "how much chance it wil return",
        "what if item is broked",
        "can i exchng this",
    ]
    for query in unseen_questions:
        answer = _answer(query)["answer"].lower()
        for phrase in REFUSAL_PHRASES:
            assert phrase not in answer, f"{query!r} triggered a blanket refusal"


@pytest.mark.parametrize("query,expected_fine_intent", [
    ("Hi", "greeting"),
    ("Hello, what's up?", "greeting"),
    ("What can you do?", "general_help"),
    ("Can you tell me a bedtime story?", "unsupported"),
])
def test_conversational_questions_are_not_refused_or_treated_as_policy(query, expected_fine_intent):
    result = run_once(query)
    assert result["intent"] == "conversational"
    assert result["fine_intent"] == expected_fine_intent
    assert result["response"]["source"] == "conversational"
    answer = result["response"]["answer"].lower()
    assert "can't answer that from flipkart's policy knowledge base" not in answer


# --------------------------------------------------------- typos / informality
def test_typo_and_informal_phrasing_still_reaches_a_grounded_policy_answer():
    result = run_once("can i retun shoes after 8 day")
    assert result["intent"] == "policy"
    assert result["groundedness"]["grounded"] is True
    assert "footwear" in result["response"]["answer"].lower()


def test_informal_delivery_complaint_is_understood_as_policy_not_refused():
    result = run_once("my order came late what can i do")
    assert result["intent"] == "policy"
    assert result["response"]["source"] == "policy_kb"


# ------------------------------------------------------------ product catalog
def test_product_catalog_question_is_answered_from_catalog_records():
    result = run_once("Are headphones returnable?")
    assert result["fine_intent"] == "product_lookup"
    assert result["response"]["source"] == "product_catalog"
    assert "headphones" in result["response"]["answer"].lower()


def test_structured_product_filter_question_uses_filter_products():
    result = run_once("Which products support cash on delivery?")
    assert result.get("product_hits"), "expected catalog hits for a filter-style question"
    assert all(hit["product"]["cod_available"] for hit in result["product_hits"])
    assert result["response"]["source"] == "product_catalog"


def test_non_returnable_products_are_correctly_filtered():
    result = run_once("Show me products that are non-returnable.")
    assert result.get("product_hits")
    assert all(hit["product"]["non_returnable"] for hit in result["product_hits"])


# --------------------------------------------------------------- comparisons
def test_comparison_question_pulls_more_than_one_document():
    result = run_once("Can you compare the return policies for electronics and footwear?")
    assert result["fine_intent"] == "comparison"
    assert len({h["document_id"] for h in result["doc_hits"]}) >= 2
    answer = result["response"]["answer"]
    assert "Comparing" in answer or "compar" in answer.lower()


# -------------------------------------------------------- conversation memory
def test_follow_up_pronoun_is_resolved_against_the_last_topic():
    conversation = Conversation("test-followup")
    first = conversation.ask("What is the return window for shoes?")
    assert first.get("last_topic")
    second = conversation.ask("What about if they are damaged?")
    assert second["fine_intent"] == "damaged_product"
    assert first["last_topic"] in second["retrieval_query"]


def test_order_slots_accumulate_across_turns_without_an_order_id():
    conversation = Conversation("test-slots")
    conversation.ask("I bought a pair of running shoes for ₹4,500 using COD.")
    second = conversation.ask("It arrived 6 days late. What is the return risk?")

    features = second["order_features"]
    assert features["price_inr"] == 4500.0
    assert features["payment_method"] == "COD"
    assert features["product_category"] == "Footwear"
    assert features["delivery_days"] == 6

    # Still short of the 11 features the model needs -- the agent must ask by
    # name for exactly what's missing, not guess and not give a blanket refusal.
    answer = second["response"]["answer"]
    assert second["response"]["source"] == "return_risk_tool"
    assert second["tool_result"]["status"] == "missing_input"
    assert "discount" in answer.lower()


# ------------------------------------------------------- schema stays honest
def test_all_new_lanes_still_match_the_fixed_json_schema():
    queries = [
        "Hi",
        "What can you do?",
        "Are headphones returnable?",
        "Can you compare footwear and electronics returns?",
        "Can you explain the refund policy in simple terms?",
    ]
    for query in queries:
        response = _answer(query)
        assert set(response) == {"answer", "source", "confidence"}
        assert response["source"] in VALID_SOURCES
        assert 0.0 <= response["confidence"] <= 1.0
        json.dumps(response)


def test_multi_clause_question_answers_from_real_retrieved_evidence():
    result = run_once(
        "I bought headphones using COD, they arrived 4 days late and I opened "
        "the box. Can I return them and what happens to my refund?"
    )
    assert result["intent"] == "policy"
    if result["groundedness"]["grounded"]:
        best_chunk = result["doc_hits"][0]["best_chunk_text"]
        assert best_chunk in result["response"]["answer"]
    else:
        assert f"{result['groundedness']['best_score']:.4f}" in result["response"]["answer"]
