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
def assistant(state: MessagesState):
    return {'messages': [llm_with_tools.invoke([sys_msg] + state["messages"])]}

class State(MessagesState):
    pass


workflow = StateGraph(State)

# Define nodes: these do the work
workflow.add_node("assistant", assistant)
workflow.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
workflow.add_edge(START, "assistant")
workflow.add_conditional_edges("assistant", tools_condition)
workflow.add_edge("tools", "assistant")
memory = MemorySaver()
graph = workflow.compile(interrupt_before=["assistant"],checkpointer=memory)
thread = {"configurable": {"thread_id": "1234"}}
# %%
# Show
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

# %%
for event in graph.stream({"messages": [HumanMessage(content="Add 3 and 4.")]}, thread, stream_mode='values'):
    event['messages'][-1].pretty_print()

# %%
state = graph.get_state(thread)
state.next
# %%
graph.update_state(thread, {"messages": [HumanMessage(content="No. Multiply 3 and 4.")]})
# %%
for event in graph.stream(None, thread, stream_mode='values'):
    event['messages'][-1].pretty_print()
# %%
# user_approval = input('Do you want to continue? yes or no')
# if user_approval == 'yes':
#     for event in graph.stream(None, thread, stream_mode='values'):
#         event['messages'][-1].pretty_print()
# else:
#     print('User does not want to continue')

# %%