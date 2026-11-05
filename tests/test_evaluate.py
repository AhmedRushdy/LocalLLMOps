
from evaluation.evaluate import (
    evaluate_case,
    count_numbered_items,
    count_bullet_items,
)


def test_required_phrase_passes():
    test = {
        "must_include_all": ["large language model operations"]
    }
    result = evaluate_case(
        test,
        "LLMOps means large language model operations.",
    )
    assert result["passed"] is True


def test_missing_required_phrase_fails():
    test = {
        "must_include_all": ["large language model operations"]
    }
    result = evaluate_case(
        test,
        "LLMOps is a way to optimize models.",
    )
    assert result["passed"] is False


def test_alternative_phrase_passes():
    test = {"must_include_any": ["prediction", "output"]}
    result = evaluate_case(
        test,
        "The model produces an output.",
    )
    assert result["passed"] is True


def test_exactly_three_numbered_items_pass():
    response = "1. First\n2. Second\n3. Third"
    test = {
        "min_numbered_items": 3,
        "max_numbered_items": 3,
    }
    result = evaluate_case(test, response)
    assert result["passed"] is True
    assert result["numbered_item_count"] == 3


def test_four_numbered_items_fail():
    response = "1. First\n2. Second\n3. Third\n4. Fourth"
    test = {
        "min_numbered_items": 3,
        "max_numbered_items": 3,
    }
    result = evaluate_case(test, response)
    assert result["passed"] is False


def test_bullets_and_numbered_items_are_separate():
    response = "1. Numbered\n- Bullet\n* Another bullet"
    assert count_numbered_items(response) == 1
    assert count_bullet_items(response) == 2
