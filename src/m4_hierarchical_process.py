"""
04 - Hierarchical Process
===========================
`Process.sequential` runs a fixed list of tasks in order.
`Process.hierarchical` is different: you give the crew a pool of
worker agents and a manager (either `manager_agent=` or
`manager_llm=`), and the manager decides at runtime who does what,
using CrewAI's built-in "Delegate work to coworker" /
"Ask question to coworker" tools -- the same live Action/Action Input
loop documented in docs/react_transcript_example.md, just aimed at
other agents instead of a data tool.

That live delegation decision is genuine reasoning a deterministic
stand-in can't meaningfully fake -- unlike the other tutorial modules,
this one builds and validates a real hierarchical Crew, but only calls
`.kickoff()` if a real LLM is available (`ANTHROPIC_API_KEY` set). See
crewai-manufacturing-quality-crew for this same trade-off made
explicit at application scale: sequential by default (offline-safe),
hierarchical behind a flag that requires a real key.

Run:
    python src/m4_hierarchical_process.py
    ANTHROPIC_API_KEY=... python src/m4_hierarchical_process.py   # actually delegates
"""
from _common import banner  # noqa: F401

from crewai import Agent, Task, Crew, Process

from offline_llm import DeterministicLLM, real_llm


class _SpecialistLLM(DeterministicLLM):
    """Concrete placeholder for a worker agent's brain -- not the point
    of this module (the manager's delegation decision is)."""

    def respond(self, role: str, task: str, context: str) -> str:
        return f"[{role}] would diagnose the issue here given a real LLM."


def build_crew(manager_llm) -> Crew:
    vibration_specialist = Agent(
        role="Vibration Specialist",
        goal="Diagnose rotating-equipment vibration issues",
        backstory="You are an expert in rotating machinery vibration analysis.",
        llm=_SpecialistLLM(),
        verbose=False,
        allow_delegation=False,
    )
    thermal_specialist = Agent(
        role="Thermal Specialist",
        goal="Diagnose overheating and thermal-stress issues",
        backstory="You are an expert in industrial thermal analysis.",
        llm=_SpecialistLLM(),
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "A compressor is showing both elevated vibration and elevated "
            "bearing temperature. Determine which specialist should lead "
            "the investigation and produce a diagnosis."
        ),
        expected_output="A diagnosis with a named lead specialist.",
    )

    return Crew(
        agents=[vibration_specialist, thermal_specialist],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=manager_llm,
        verbose=False,
        tracing=False,
    )


def main() -> None:
    banner("Hierarchical process (manager delegates to workers)")

    # A deterministic LLM can't play the manager role meaningfully -- picking
    # which specialist should own an ambiguous, overlapping symptom is a
    # judgment call, not a lookup -- but it's enough to prove the crew is
    # constructed correctly.
    crew = build_crew(manager_llm=_SpecialistLLM())
    print(f"Process:        {crew.process}")
    print(f"Worker agents:  {[a.role for a in crew.agents]}")
    print("Hierarchical crew constructed successfully.\n")

    try:
        llm = real_llm()
    except RuntimeError as e:
        print(f"Skipping kickoff(): {e}")
        print("Rebuild the crew with manager_llm=real_llm() to see live delegation.")
        return

    crew = build_crew(manager_llm=llm)
    result = crew.kickoff()
    print(f"Result: {result.raw}")


if __name__ == "__main__":
    main()
