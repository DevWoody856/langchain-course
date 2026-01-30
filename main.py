from typing import Any
import os
from pprint import pformat

from dotenv import load_dotenv
from langchain_tavily import TavilySearch



load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


tools = [TavilySearch(include_domains=["linkedin.com"])]
llm = ChatOpenAI(
    model="z-ai/glm-4.5-air:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0,
    model_kwargs={"tool_choice": "auto"},
)
chain = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant. Use at most one tool call if needed. "
        "If a tool errors, stop and answer with what you have."
    ),
)


def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        {"messages": [{"role": "user", "content": "search for 3 jobs for an ai enginner using langchain in the bay area on linkedin and list their details"}]},
        config={"recursion_limit": 8},
    )
    messages = result.get("messages", []) if isinstance(result, dict) else []
    last_message = messages[-1] if messages else None
    if last_message is not None and hasattr(last_message, "content"):
        print("\n=== Answer ===")
        print(last_message.content)
    else:
        print("\n=== Answer ===")
        print(pformat(result, width=100))

    print("\n=== Debug (messages summary) ===")
    summary: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            summary.append(msg)
            continue
        item: dict[str, Any] = {"type": msg.__class__.__name__}
        if hasattr(msg, "type"):
            item["type"] = getattr(msg, "type")
        if hasattr(msg, "content"):
            item["content"] = getattr(msg, "content")
        if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls"):
            item["tool_calls"] = getattr(msg, "tool_calls")
        summary.append(item)
    print(pformat(summary, width=100))


if __name__ == "__main__":
    main()
