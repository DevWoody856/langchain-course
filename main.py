from typing import List
import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


load_dotenv()

from langchainhub import hub
from langchain.agents import AgentExecutor, create_agent
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class Source(BaseModel):
    """ Schema for a source used by the agent."""

    url:str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """ Schema for agent response with answer and sources."""

    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

tools = [TavilySearch()]
llm = ChatOpenAI(
    model="z-ai/glm-4.5-air:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0,
    model_kwargs={"tool_choice": "auto"},
)

react_prompt = hub.pull("hwchase17/react-agent")
agent = create_react_agent(
    model=llm, 
    tools=tools,
    prompt=react_prompt
    )
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        input={"input": "What are the latest advancements in AI technology?"}
    )
    print(result)


if __name__ == "__main__":
    main()
