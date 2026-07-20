"""
A deterministic, zero-API-key stand-in LLM for CrewAI agents.

CrewAI drives an agent by repeatedly calling `llm.call(messages, ...)`
with a growing conversation: a system message describing the agent's
role/goal (and, if the agent has tools, a ReAct-style
"Thought / Action / Action Input / Observation" protocol description),
and a user message with the current task, any upstream context from
earlier tasks in the crew (via `Task(context=[...])`), and prior
Observations. A response containing "Final Answer:" ends the loop; the
text after it becomes the task's output.

`DeterministicLLM` implements CrewAI's `BaseLLM` interface (the same
extension point you'd use for any custom or self-hosted model -- see
https://docs.crewai.com for the abstract contract) but always answers
in a single turn: it parses the agent's role and task text out of the
prompt with `extract_role`/`extract_current_task`/`extract_context`,
and delegates to a subclass-supplied `respond(role, task, context)`
function to produce the Final Answer's content. That keeps every demo
in this repo runnable with **no API key**, while making the swap to a
real model a one-line change -- see `real_llm()` below.

This module also documents (and its tests exercise) exactly what a
real LLM's ReAct transcript looks like when an agent actually invokes a
tool, captured verbatim from a live crewai run -- see
docs/react_transcript_example.md.
"""
import os
import re
from abc import abstractmethod

from crewai import BaseLLM

_ROLE_RE = re.compile(r"You are (.*?)\.")
_TASK_RE = re.compile(
    r"Current Task:\s*(.*?)\n\nThis is the expected criteria", re.DOTALL
)
_CONTEXT_RE = re.compile(
    r"This is the context you're working with:\n(.*?)\n\nProvide your complete response",
    re.DOTALL,
)


def extract_role(prompt_text: str) -> str:
    m = _ROLE_RE.search(prompt_text)
    return m.group(1) if m else "Agent"


def extract_current_task(prompt_text: str) -> str:
    m = _TASK_RE.search(prompt_text)
    return m.group(1).strip() if m else ""


def extract_context(prompt_text: str) -> str:
    """Text from upstream tasks, populated by CrewAI when a Task is
    created with `context=[other_task]` in a sequential crew."""
    m = _CONTEXT_RE.search(prompt_text)
    return m.group(1).strip() if m else ""


class DeterministicLLM(BaseLLM):
    """Base class for a rule-based `BaseLLM`. Subclasses implement
    `respond()`; this class handles the CrewAI plumbing (message
    flattening, role/task/context extraction, and formatting the
    result as a ReAct "Final Answer")."""

    def __init__(self, model: str = "offline/deterministic"):
        super().__init__(model=model)

    @abstractmethod
    def respond(self, role: str, task: str, context: str) -> str:
        """Return the content of the Final Answer for this role/task."""

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        text = (
            messages
            if isinstance(messages, str)
            else " ".join(str(m.get("content", "")) for m in messages)
        )
        role = extract_role(text)
        task = extract_current_task(text)
        context = extract_context(text)

        answer = self.respond(role, task, context)
        return f"Thought: I now know the final answer\nFinal Answer: {answer}"


def real_llm():
    """Build a genuine LLM-backed crewai.LLM, used only if
    ANTHROPIC_API_KEY is set. Swapping this in for a DeterministicLLM
    is the only change needed to make any crew in this repo a real
    reasoning agent -- Agents/Tasks/Tools/Crews are untouched, because
    they only depend on the BaseLLM interface, not on which
    implementation of it they're given.
    """
    from crewai import LLM

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY to use a real model instead of DeterministicLLM."
        )
    return LLM(model="anthropic/claude-sonnet-5")
