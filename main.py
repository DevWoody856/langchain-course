from typing import List
import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class Source(BaseModel):
    """ Schema for a source used by the agent."""

    url:str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """ Schema for agent response with answer and sources."""

    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


llm = ChatOpenAI(
    model="z-ai/glm-4.5-air:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0,
    model_kwargs={"tool_choice": "auto"},
)

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="search for 3 job postings for an ai engineer using langchain in the day area on linkedin"
                )
            ]
        },
        config={"recursion_limit": 100},
    )
    print(result)


if __name__ == "__main__":
    main()
