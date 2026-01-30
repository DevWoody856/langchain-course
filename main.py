from typing import Any
import os
from pprint import pformat

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers.pydantic import PydanticOutputParser

from schemas import AgentResponse

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


tools = [TavilySearch(include_domains=["linkedin.com"])]
llm = ChatOpenAI(
    model="upstage/solar-pro-3:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0,
    model_kwargs={"tool_choice": "auto"},
)
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
format_instructions = output_parser.get_format_instructions()


agent_chain = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant. Use at most one tool call if needed. "
        "If a tool errors, stop and answer with what you have."
    ),
    interrupt_after=["tools"],
)

def _build_structured_prompt(result: dict[str, Any]) -> list:
    messages = result.get("messages", []) if isinstance(result, dict) else []
    tool_messages = [msg for msg in messages if getattr(msg, "type", None) == "tool"]
    tool_context = tool_messages[-1].content if tool_messages else ""
    user_messages = [msg for msg in messages if getattr(msg, "type", None) == "human"]
    question = user_messages[-1].content if user_messages else ""
    prompt = (
        "Answer the user's question using the tool results if available. "
        "If tool results are empty, answer based on your general knowledge.\n\n"
        f"Question: {question}\n\n"
        f"Tool results:\n{tool_context}\n\n"
        "Return the answer in the following JSON format:\n"
        f"{format_instructions}"
    )
    return [SystemMessage(content="You are a helpful assistant."), HumanMessage(content=prompt)]


extract_output = RunnableLambda(_build_structured_prompt)

def _parse_or_fallback(text: str) -> AgentResponse:
    try:
        return output_parser.parse(text)
    except Exception:
        return AgentResponse(answer=text, sources=[])


parse_output = RunnableLambda(_parse_or_fallback)


def main():
    print("Hello from langchain-course!")
    result = agent_chain.invoke(
        {"messages": [{"role": "user", "content": "search for 3 jobs for an ai enginner using langchain in the bay area on linkedin and list their details"}]},
    )
    prompt_messages = extract_output.invoke(result)
    final_response = llm.invoke(prompt_messages)
    parsed = parse_output.invoke(final_response.content if hasattr(final_response, "content") else "")
    messages = result.get("messages", []) if isinstance(result, dict) else []
    print("\n=== Answer ===")
    print(parsed.answer)
    if parsed.sources:
        print("\n=== Sources ===")
        for source in parsed.sources:
            print(f"- {source.url}")
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
