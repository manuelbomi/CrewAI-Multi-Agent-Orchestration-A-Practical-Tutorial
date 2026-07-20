"""
01 - Agents, Tasks, and Crews
==============================
The three building blocks of CrewAI:

  - an Agent has a role, a goal, and a backstory -- these aren't
    flavor text, they get compiled directly into the system prompt
    that steers the model's behavior for every task it's given.
  - a Task has a description and an expected_output, and is assigned
    to exactly one Agent.
  - a Crew bundles agents + tasks + a Process (how tasks are run) and
    is the thing you actually call `.kickoff()` on.

This module runs the smallest possible crew: one agent, one task.

Run:
    python src/m1_agents_tasks_crews.py
"""
from _common import banner  # noqa: F401 (import first: sets CREWAI_TRACING_ENABLED=false)

from crewai import Agent, Task, Crew, Process

from offline_llm import DeterministicLLM


class SummarizerLLM(DeterministicLLM):
    def respond(self, role: str, task: str, context: str) -> str:
        return (
            f"As the {role}, here is a one-sentence summary of the task "
            f"I was given: '{task[:80]}...'. This response was generated "
            "deterministically -- see offline_llm.py -- to keep this repo "
            "runnable with no API key."
        )


def main() -> None:
    banner("Agents, Tasks, and Crews")

    llm = SummarizerLLM()

    analyst = Agent(
        role="Research Analyst",
        goal="Produce clear, concise summaries of technical topics",
        backstory=(
            "You are a research analyst at an energy company, known for "
            "distilling complex technical material into plain language "
            "for non-specialist stakeholders."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=(
            "Summarize, in one sentence, why formal ontologies help ground "
            "AI agents in facts instead of hallucinated answers."
        ),
        expected_output="One clear sentence.",
        agent=analyst,
    )

    crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False, tracing=False)
    result = crew.kickoff()

    print(f"Agent role:        {analyst.role}")
    print(f"Task description:  {task.description}")
    print(f"Crew result:       {result.raw}")


if __name__ == "__main__":
    main()
