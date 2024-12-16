# %%
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
import random
from pprint import pprint
# from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage
load_dotenv()
from langgraph.graph import StateGraph, START, END
from typing import Literal
from IPython.display import Image, display
# %%

class State(TypedDict):
    graph_state: str

def node1(state):
    print("---Node 1---")
    return {'graph_state': state['graph_state'] + ", I am"}

def node2(state):
    print("---Node 2---")
    return {'graph_state': state['graph_state'] + " happy"}

def node3(state):
    print("---Node 3---")
    return {'graph_state': state['graph_state'] + " sad"}

# def addtool(a: int, b: int) -> int:
#     print("---Add Tool---")
#     return a + b

def moodset(state) -> Literal["node2", "node3"]:
    print("---Mood Set---")
    if random.random() < 0.5:
        return "node2"
    return "node3"
# %%
# chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# # %%
# mesg = [HumanMessage(content="What is 2 + 5",name="Sandeep"),
#             AIMessage(content="Hi",name="AI")]
# chat_llm_withtools = chat_llm.bind_tools([addtool])
# print(chat_llm_withtools.invoke(mesg))
# %%
builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("node3", node3)

builder.add_edge(START, "node1")
builder.add_conditional_edges("node1", moodset)
builder.add_edge("node2", END)
builder.add_edge("node3", END)
graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))
# View
# %%
graph.invoke({'graph_state': "I am Sandeep"})
# %%
