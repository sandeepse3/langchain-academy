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

# llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
# %%
# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine the control flow
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(checkpointer=MemorySaver())

# Show
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
# %%
# Input
initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

# %%
graph.get_state({'configurable': {'thread_id': '1'}})
print(graph.get_state({'configurable': {'thread_id': '1'}}).next)
# %%
all_states = [s for s in graph.get_state_history(thread)]
len(all_states)
# %%
# The first element is the current state
all_states[0].next
# %%
# Replay - go back to the previous state and starts from there without executing the Nodes
to_replay = all_states[-2]
print(to_replay)
to_replay.next
# %%
to_replay.values
# %%
to_replay.config
# %%
# Pass new checkpoint along with the thread
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    event['messages'][-1].pretty_print()
# %%
all_states = [s for s in graph.get_state_history(thread)]
len(all_states)
# %%
# Forking - If we want to run from that same step, but with a different input
to_fork = all_states[-2]
to_fork.values["messages"]
# %%
to_fork.config
# %%
# New Snapshot / Checkpoint with new input at that ID
fork_config = graph.update_state(
    to_fork.config,
    {"messages": [HumanMessage(content='Multiply 5 and 3', 
                               id=to_fork.values["messages"][0].id)]},
)
# %%
fork_config
# %%
all_states = [state for state in graph.get_state_history(thread)]
all_states[0].values["messages"]
# %%
graph.get_state({'configurable': {'thread_id': '1'}})
# %%
# Pass new checkpoint along with the thread
for event in graph.stream(None, fork_config, stream_mode="values"):
    event['messages'][-1].pretty_print()
# %%
graph.get_state({'configurable': {'thread_id': '1'}})
# %%
graph.get_state({'configurable': {'thread_id': '1'}}).next