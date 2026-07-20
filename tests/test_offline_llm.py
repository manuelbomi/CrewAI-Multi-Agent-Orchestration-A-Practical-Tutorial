from offline_llm import extract_role, extract_current_task, extract_context


def test_extract_role():
    text = "You are Research Analyst. Your personal goal is: do things"
    assert extract_role(text) == "Research Analyst"


def test_extract_role_defaults_when_absent():
    assert extract_role("no role marker here") == "Agent"


def test_extract_current_task():
    text = (
        "Current Task: Summarize the report\n\n"
        "This is the expected criteria for your final answer: one sentence."
    )
    assert extract_current_task(text) == "Summarize the report"


def test_extract_context_present():
    text = (
        "This is the context you're working with:\n"
        "prior agent's output here\n\n"
        "Provide your complete response: go"
    )
    assert extract_context(text) == "prior agent's output here"


def test_extract_context_absent():
    assert extract_context("no context section here") == ""
