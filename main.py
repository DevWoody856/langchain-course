import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from schemas import AgentResponse
from langchain_tavily import TavilySearch


llm = ChatOpenAI(
    model="z-ai/glm-4.5-air:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0,
)

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {"messages": [HumanMessage(content="What's the weather like in Tokyo?")]}
    )
    print(result)


if __name__ == "__main__":
    main()
