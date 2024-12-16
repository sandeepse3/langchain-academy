# %%
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
import random
from pprint import pprint
from langchain_core.messages import AnyMessage, SystemMessage,HumanMessage,AIMessage,ToolMessage
load_dotenv()
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

# %%
# React Agent - Reason Act Observe
# act - let the model call specific tools
# observe - pass the tool output back to the model
# reason - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)

# This will be a tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# %%
# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def tool_call_func(state: MessagesState):
    return {'messages': [llm_with_tools.invoke([sys_msg] + state["messages"])]}

class State(MessagesState):
    pass


workflow = StateGraph(State)

# Define nodes: these do the work
workflow.add_node("tool_call_func", tool_call_func)
workflow.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
workflow.add_edge(START, "tool_call_func")
workflow.add_conditional_edges("tool_call_func", tools_condition)
workflow.add_edge("tools", "tool_call_func")
memory = MemorySaver()
react_graph = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "1234"}}
# %%

messages = react_graph.invoke({"messages": [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]}, config)
for m in messages['messages']:
    m.pretty_print()
# %%
messages = react_graph.invoke({"messages": [HumanMessage(content="Multiply that by 10")]}, config)
for m in messages['messages']:
    m.pretty_print()
# %%
messages = react_graph.invoke({"messages": [HumanMessage(content="Add that by 10")]}, config)
for m in messages['messages']:
    m.pretty_print()
# %%
