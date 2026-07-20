import os

# Prevents CrewAI's interactive "view execution traces?" prompt (which
# otherwise blocks for up to 20s waiting on stdin) from firing when
# these scripts are run non-interactively. Pass tracing=False to every
# Crew(...) as well -- see each module for that.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")


def banner(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")
