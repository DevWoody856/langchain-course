from typing import Any, List, Union
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from callbacks import AgentCallbackHandler

load_dotenv()


class ReActSingleInputOutputParser:
    def __call__(self, output: Any) -> Union[AgentAction, AgentFinish]:
        text = output.content if hasattr(output, "content") else str(output)
        if "Final Answer:" in text:
            final_answer = text.split("Final Answer:", 1)[1].strip()
            return AgentFinish(return_values={"output": final_answer}, log=text)

        action_match = re.search(r"Action:\s*(.+)", text)
        action_input_match = re.search(r"Action Input:\s*(.+)", text)
        if not action_match:
            return AgentFinish(return_values={"output": text.strip()}, log=text)

        tool_name = action_match[1].strip()
        tool_input = action_input_match[1].strip() if action_input_match else ""
        return AgentAction(tool=tool_name, tool_input=tool_input, log=text)

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  #
    return len(text)

def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


def format_log_to_str(steps: List[tuple[AgentAction, str]]) -> str:
    lines: list[str] = []
    for action, observation in steps:
        lines.append(action.log)
        lines.append(f"Observation: {observation}")
    return "\n".join(lines)


def render_tool_descriptions(tools: List[BaseTool]) -> str:
    lines: list[str] = []
    for tool in tools:
        description = getattr(tool, "description", None) or (tool.func.__doc__ or "")
        lines.append(f"{tool.name}: {description.strip()}")
    return "\n".join(lines)


llm = ChatOpenAI(
    model="upstage/solar-pro-3:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    temperature=0,
    model_kwargs={"tool_choice": "auto"},
    callbacks=[AgentCallbackHandler()],
)
intermediate_steps = []

if __name__ == "__main__":
    print("Hello React Langchain!")
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}.
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_tool_descriptions(tools),
        tool_names=", ".join([tool.name for tool in tools])
    )

    agent = (
        {"input": lambda x: x["input"], 
         "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
         } 
         | prompt 
         | llm 
         | ReActSingleInputOutputParser()
             )

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": 'What is the length of the text "Hello, world!"?',
            "agent_scratchpad": intermediate_steps,
            }
        )
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
        intermediate_steps.append((agent_step, observation))