# %%
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
import random
from pprint import pprint
from langgraph.graph import StateGraph, START, END, MessagesState
load_dotenv()
from langchain_core.messages import AnyMessage, SystemMessage,HumanMessage,AIMessage,ToolMessage
from langgraph.graph.message import add_messages
from IPython.display import Image, display
# %%
messages = [AIMessage(content=f"So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content=f"Yes, that's right.",name="Lance"))
messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US.", name="Lance"))

for m in messages:
    m.pretty_print()

# %%
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

def subtract(a: int, b: int) -> int:
    """Subtract a and b.

    Args:
        a: first int
        b: second int
    """
    return a - b
# %%
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([multiply, divide, subtract])

# %%
class State(MessagesState):
    pass

# %%
def aiassistant(state:State):
    return {"messages": [llm_with_tools.invoke(state['messages'])]}
workflow = StateGraph(State)
workflow.add_node("aiassistant",aiassistant)
workflow.add_edge(START,"aiassistant")
workflow.add_edge("aiassistant",END)
graph = workflow.compile()
# %%
print('------------------------------')
res = graph.invoke({"messages": [HumanMessage(content="Multiply 2 and 3")]})
for m in res['messages']:
    m.pretty_print()
