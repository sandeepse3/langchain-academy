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
from IPython.display import Image, display
# %%
# messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
# messages.append(HumanMessage(content=f"Yes, that's right.",name="Lance"))
# messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
# messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Lance"))

# for m in messages:
#     m.pretty_print()
# %%
llm = ChatOpenAI(model="gpt-4o")
# result = llm.invoke(messages)
# type(result)
# result
# %%
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b
llm_with_tools = llm.bind_tools([multiply])
# %%
# tool_call = llm_with_tools.invoke([HumanMessage(content=f"What is 2 multiplied by 3", name="Lance")])
# print(tool_call)
# print(tool_call.additional_kwargs['tool_calls'])
# %%
# class MessagesState(TypedDict):
#     messages: list[AnyMessage]
# # Initial state
# initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
#                     HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
#                    ]

# # New message to add
# new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

# Test
# mesgs = add_messages(initial_messages , new_message)
# for m in mesgs:
#     m.pretty_print()
# %%
def tool_call_func(state: MessagesState):
    return {'messages': [llm_with_tools.invoke(state["messages"])]}

class State(MessagesState):
    pass

builder = StateGraph(State)
builder.add_node("tool_call_func", tool_call_func)
builder.add_edge(START, "tool_call_func")
builder.add_edge("tool_call_func", END)
graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
# View
# %%
messages = graph.invoke({"messages": HumanMessage(content="Hello!")})
for m in messages['messages']:
    m.pretty_print()
# %%
messages = graph.invoke({"messages": HumanMessage(content="Multiply 2 and 3")})
for m in messages['messages']:
    m.pretty_print()