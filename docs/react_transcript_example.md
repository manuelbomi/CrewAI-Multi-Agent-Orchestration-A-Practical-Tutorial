# What a real LLM's tool-calling transcript looks like

Everything in this repo runs by default against `DeterministicLLM`
(see `src/offline_llm.py`), which never uses CrewAI's live
tool-calling loop -- it answers in one turn. This document captures
what CrewAI actually sends a *real* tool-calling model, verbatim, from
a live run, so the gap between "demo mode" and "production mode" is
explicit rather than hand-waved.

## The system prompt CrewAI builds for a tool-using agent

When an `Agent` has `tools=[...]`, CrewAI injects a ReAct-style
protocol description into the system message automatically:

```
You are Tester. A minimal test agent.
Your personal goal is: Call the tool and report its result
You ONLY have access to the following tools, and should NEVER make up tools that are not listed here:

Tool Name: get_number
Tool Arguments: {
  "properties": {},
  "title": "EchoToolSchema",
  "type": "object",
  "additionalProperties": false,
  "required": []
}
Tool Description: Returns a fixed number for testing.

IMPORTANT: Use the following format in your response:

```
Thought: you should always think about what to do
Action: the action to take, only one name of [get_number], just the name, exactly as it's written.
Action Input: the input to the action, just a simple JSON object, enclosed in curly braces, using " to wrap keys and values.
Observation: the result of the action
```

Once all necessary information is gathered, return the following format:

```
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```
```

## The loop

1. CrewAI calls `llm.call(messages)` with that system prompt plus the
   task description.
2. A real model replies with `Thought: ... / Action: get_number /
   Action Input: {}`.
3. CrewAI's executor parses the `Action`/`Action Input`, calls the
   matching tool itself, and appends `Observation: <tool's return
   value>` to the conversation.
4. CrewAI calls `llm.call(messages)` again with the grown transcript.
5. The model now replies with `Thought: I now know the final answer /
   Final Answer: ...`, which ends the loop and becomes the task's
   output.

`DeterministicLLM` collapses steps 2-5 into a single turn: rather than
emitting an `Action` and waiting for CrewAI to run the tool, it
extracts whatever data was pre-fetched and embedded directly into the
task description (see `m3_custom_tools.py` and the two applied repos'
`src/crew.py`) and answers immediately. This sidesteps needing to
reverse-engineer CrewAI's exact `Action`/`Observation` parsing (which
is an internal implementation detail, not a stable public contract)
while still producing an answer grounded in real tool output rather
than invented text.

Swap in `offline_llm.real_llm()` (requires `ANTHROPIC_API_KEY`) and
the *exact* transcript above is what actually drives every agent in
this repo -- nothing else about the Agent/Task/Tool/Crew code changes.
