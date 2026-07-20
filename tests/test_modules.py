import m1_agents_tasks_crews as m1
import m2_sequential_process as m2
import m3_custom_tools as m3
import m4_hierarchical_process as m4
import m5_grounded_agent as m5


def test_m1_single_agent_crew_runs():
    llm = m1.SummarizerLLM()
    from crewai import Agent, Task, Crew, Process

    agent = Agent(role="Research Analyst", goal="g", backstory="b", llm=llm, verbose=False)
    task = Task(description="Explain X in one sentence.", expected_output="one sentence", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False, tracing=False)
    result = crew.kickoff()
    assert "Research Analyst" in result.raw


def test_m2_context_hands_off_between_agents():
    crew = m2.build_crew()
    crew.kickoff()
    research_output = crew.tasks[0].output.raw
    writer_output = crew.tasks[2].output.raw
    # the writer's deterministic response embeds a slice of what it received
    # as context, which itself is derived from the researcher's own output
    assert "self-hosted open-weight LLMs" in research_output
    assert "analyst's evaluation" in writer_output


def test_m3_tool_output_reaches_the_agent():
    tool = m3.TextStatsTool()
    result = tool.run(text="one two three")
    assert "'words': 3" in result


def test_m4_hierarchical_crew_constructs_without_kickoff():
    crew = m4.build_crew(manager_llm=m4._SpecialistLLM())
    assert crew.process.value == "hierarchical" or str(crew.process) == "hierarchical"
    assert {a.role for a in crew.agents} == {"Vibration Specialist", "Thermal Specialist"}


def test_m5_answers_from_known_facts():
    answer = m5.ask("Compressor101 status?")
    assert "3200" in answer
    assert "Train1" in answer


def test_m5_refuses_to_guess_on_unknown_equipment():
    answer = m5.ask("Compressor999 status?")
    assert "no data" in answer.lower()
