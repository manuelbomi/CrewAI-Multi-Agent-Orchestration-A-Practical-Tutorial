"""
05 - Grounded Agents (capstone)
=================================
The pattern every other module has been building toward: an agent that
answers *only* from facts it was actually given, and says so plainly
when a question falls outside them -- instead of quietly filling the
gap with something plausible-sounding.

`FACTS` here is a tiny stand-in "knowledge source" (three facts about a
compressor). In the two applied repos in this portfolio
(crewai-lng-maintenance-crew, crewai-manufacturing-quality-crew), this
exact role is played by a real OWL/RDF knowledge graph queried through
a CrewAI tool -- same contract, real data, industrial scale.

Run:
    python src/m5_grounded_agent.py
"""
import ast

from _common import banner  # noqa: F401

from crewai import Agent, Task, Crew, Process

from offline_llm import DeterministicLLM

FACTS = {
    "Compressor101": {"train": "Train1", "rated_power_kw": 3200, "vibration_mm_s": 6.3},
}


class GroundedAnalystLLM(DeterministicLLM):
    def respond(self, role: str, task: str, context: str) -> str:
        if "FACTS::" not in task:
            return "No facts were provided; I cannot answer."
        facts_blob = task.split("FACTS::", 1)[1].split("::ENDFACTS", 1)[0]
        question = task.split("QUESTION::", 1)[1].split("::ENDQUESTION", 1)[0]

        facts = ast.literal_eval(facts_blob)
        equipment = question.strip().split()[0]

        if equipment not in facts:
            return f"I have no data on {equipment} in the facts I was given -- I won't guess."

        record = facts[equipment]
        return (
            f"{equipment} is in {record['train']}, rated at {record['rated_power_kw']} kW, "
            f"with a latest vibration reading of {record['vibration_mm_s']} mm/s."
        )


def ask(question: str) -> str:
    analyst = Agent(
        role="Grounded Analyst",
        goal="Answer only from the facts provided, never from memory or assumption",
        backstory="You refuse to speculate about equipment you have no data on.",
        llm=GroundedAnalystLLM(),
        verbose=False,
    )
    task = Task(
        description=f"FACTS::{FACTS}::ENDFACTS QUESTION::{question}::ENDQUESTION",
        expected_output="An answer grounded only in the facts, or an explicit refusal.",
        agent=analyst,
    )
    crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False, tracing=False)
    return crew.kickoff().raw


def main() -> None:
    banner("Grounded agents: answer from facts, or say you can't")

    for question in ["Compressor101 status?", "Compressor999 status?"]:
        print(f"\nQ: {question}")
        print(f"A: {ask(question)}")


if __name__ == "__main__":
    main()
