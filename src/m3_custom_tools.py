"""
03 - Custom Tools
==================
A CrewAI `BaseTool` is just a name, a description, and a `_run()`
method -- the name/description are what get compiled into the agent's
system prompt (see docs/react_transcript_example.md for the exact
text), and `_run()` is genuine Python that executes when the model
decides to call it.

`TextStatsTool` below is deliberately simple (word/character counts)
so the mechanics are obvious. It's attached to the agent below exactly
as you would in production -- a real, tool-calling LLM would invoke it
live via the Action/Action Input loop.

For the offline default, this module uses a different, equally valid
pattern: call the tool directly in Python, and embed its result into
the Task description as the "Relevant data" the agent must answer
from. This is the same trick the two applied repos in this portfolio
(crewai-lng-maintenance-crew, crewai-manufacturing-quality-crew) use to
ground their agents' answers in a knowledge graph without requiring a
live API key -- see their src/crew.py for the pattern at real scale.

Run:
    python src/m3_custom_tools.py
"""
from _common import banner  # noqa: F401

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

from offline_llm import DeterministicLLM


class TextStatsTool(BaseTool):
    name: str = "text_stats"
    description: str = "Returns word and character counts for a piece of text."

    def _run(self, text: str) -> str:
        return f"{{'words': {len(text.split())}, 'characters': {len(text)}}}"


class StatsReporterLLM(DeterministicLLM):
    def respond(self, role: str, task: str, context: str) -> str:
        if "RELEVANT DATA::" not in task:
            return "No data was provided to summarize."
        data = task.split("RELEVANT DATA::", 1)[1].strip()
        return f"Based only on the tool's output, the text has these stats: {data}"


def main() -> None:
    banner("Custom tools + offline grounding pattern")

    tool = TextStatsTool()
    sample_text = (
        "Ontologies give agentic AI systems a shared, formal vocabulary "
        "so they answer from facts instead of guesses."
    )

    # The offline-safe grounding pattern: call the tool directly, then
    # embed its real output into the task description.
    tool_result = tool.run(text=sample_text)

    reporter = Agent(
        role="Reporter",
        goal="Report text statistics accurately",
        backstory="You report only what tools tell you, never estimates.",
        tools=[tool],  # attached for real-LLM runs; see module docstring
        llm=StatsReporterLLM(),
        verbose=False,
    )
    task = Task(
        description=(
            f"Report the word/character counts for this text: {sample_text!r}\n"
            f"RELEVANT DATA::{tool_result}"
        ),
        expected_output="A sentence stating the counts.",
        agent=reporter,
    )
    crew = Crew(agents=[reporter], tasks=[task], process=Process.sequential, verbose=False, tracing=False)
    result = crew.kickoff()

    print(f"Tool called directly:  {tool_result}")
    print(f"Agent's grounded answer: {result.raw}")


if __name__ == "__main__":
    main()
