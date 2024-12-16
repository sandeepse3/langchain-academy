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
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

# %%
llm = ChatOpenAI(model="gpt-4o")
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b
llm_with_tools = llm.bind_tools([multiply])

# %%
def tool_call_func(state: MessagesState):
    return {'messages': [llm_with_tools.invoke(state["messages"])]}

class State(MessagesState):
    pass

workflow = StateGraph(State)
workflow.add_node("tool_call_func", tool_call_func)
workflow.add_node("tools", ToolNode([multiply]))
workflow.add_edge(START, "tool_call_func")
workflow.add_conditional_edges("tool_call_func",tools_condition)
workflow.add_edge("tools", END)
graph = workflow.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))

# %%
messages = graph.invoke({"messages": [HumanMessage(content="What is 3 multiplied by 2?")]})
for m in messages['messages']:
    m.pretty_print()