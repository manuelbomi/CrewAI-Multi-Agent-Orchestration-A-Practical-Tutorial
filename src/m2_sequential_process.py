"""
02 - Sequential Process and Context Passing
=============================================
`Process.sequential` runs tasks in list order. The interesting part is
`Task(context=[other_task])`: CrewAI automatically appends the
referenced task's output into the next agent's prompt, under a
"This is the context you're working with:" section -- that's the
mechanism multi-agent hand-off is built on, and it requires no manual
prompt-stitching on your part.

This module chains three agents: a Researcher gathers a claim, an
Analyst evaluates it, and a Writer produces a final summary that can
only have come from reading the Analyst's output (each deterministic
response embeds a fingerprint of what it actually received, so the
test suite can prove the hand-off happened).

Run:
    python src/m2_sequential_process.py
"""
from _common import banner  # noqa: F401

from crewai import Agent, Task, Crew, Process

from offline_llm import DeterministicLLM


class ResearcherLLM(DeterministicLLM):
    def respond(self, role: str, task: str, context: str) -> str:
        return (
            "Claim: self-hosted open-weight LLMs reduce per-token inference "
            "cost versus hosted APIs at sustained volume, at the expense of "
            "needing to own the GPU capacity planning and serving stack."
        )


class AnalystLLM(DeterministicLLM):
    def respond(self, role: str, task: str, context: str) -> str:
        return (
            f"Evaluation of the researcher's claim ('{context[:60]}...'): "
            "directionally correct, with the crossover point depending "
            "heavily on utilization -- below ~30% GPU utilization, hosted "
            "APIs are usually still cheaper."
        )


class WriterLLM(DeterministicLLM):
    def respond(self, role: str, task: str, context: str) -> str:
        return (
            f"Final summary, built only from the analyst's evaluation "
            f"('{context[:60]}...'): self-hosting pays off at high, "
            "sustained utilization; below that, stick with hosted APIs."
        )


def build_crew() -> Crew:
    researcher = Agent(
        role="Researcher",
        goal="Surface a well-scoped technical claim",
        backstory="You research infrastructure cost trade-offs.",
        llm=ResearcherLLM(),
        verbose=False,
    )
    analyst = Agent(
        role="Analyst",
        goal="Stress-test claims for hidden assumptions",
        backstory="You are a skeptical cost analyst.",
        llm=AnalystLLM(),
        verbose=False,
    )
    writer = Agent(
        role="Writer",
        goal="Produce a crisp, decision-ready summary",
        backstory="You write for engineering leadership.",
        llm=WriterLLM(),
        verbose=False,
    )

    research_task = Task(
        description="Research whether self-hosting LLMs is cheaper than hosted APIs.",
        expected_output="One claim, one sentence.",
        agent=researcher,
    )
    analysis_task = Task(
        description="Evaluate the researcher's claim for hidden assumptions.",
        expected_output="One evaluation, one sentence.",
        agent=analyst,
        context=[research_task],
    )
    writing_task = Task(
        description="Write a final decision-ready summary from the analyst's evaluation.",
        expected_output="One summary, one sentence.",
        agent=writer,
        context=[analysis_task],
    )

    return Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=False,
        tracing=False,
    )


def main() -> None:
    banner("Sequential process with context hand-off")
    crew = build_crew()
    result = crew.kickoff()

    for task in crew.tasks:
        print(f"\n[{task.agent.role}]")
        print(f"  {task.output.raw}")

    print(f"\nFinal crew result:\n  {result.raw}")


if __name__ == "__main__":
    main()
