# Copilot instructions

## Project overview
- This repo is a step-by-step ReAct agent tutorial; the current main flow is in `main.py`.
- Architecture is two-phase: run a LangChain agent (tools + messages) then re-prompt the LLM for structured JSON and parse with Pydantic.
- Structured output schemas live in `schemas.py` (`AgentResponse`, `Source`).
- Older prompt-template patterns are kept in `prompt.py` and `ai-agent-main-1.py` for learning context.

## Key runtime flow (main pattern)
- Agent run: `create_agent(..., interrupt_after=["tools"])` with Tavily search and a strict system prompt.
- Post-process: build a follow-up prompt from agent messages, call the LLM again, parse with `PydanticOutputParser` and fall back to raw text on parse errors.
- Example: `_build_structured_prompt()` and `_parse_or_fallback()` in `main.py` show how data moves from tool output to final response.

## External services and env
- LLM uses OpenRouter via `ChatOpenAI` with `base_url="https://openrouter.ai/api/v1"` and `OPENROUTER_API_KEY`.
- Search uses Tavily via `TAVILY_API_KEY` (see README setup).
- `.env` is loaded via `python-dotenv` in scripts; keep examples aligned with README.

## Developer workflow
- Install deps with `uv sync`.
- Create `.env` from `.env.example` (see README).
- Run the agent with `uv run python main.py`.

## Conventions and patterns
- Keep prompts explicit about tool usage limits; the system prompt currently enforces at most one tool call.
- If you change schemas in `schemas.py`, update the parser usage in `main.py` and ensure format instructions stay in sync.
- For agent output parsing, preserve the fallback behavior when Pydantic parsing fails.

## Where to look
- Primary implementation: `main.py`.
- Structured output: `schemas.py`.
- Prompt templates: `prompt.py`, `ai-agent-main-1.py`.
- Tutorial context and commands: `README.md`.
