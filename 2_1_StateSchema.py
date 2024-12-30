# %%
from dotenv import load_dotenv
from typing import TypedDict
from dataclasses import dataclass
import random
# from pydantic import BaseModel, validator, ValidationError
from pydantic import BaseModel, field_validator, ValidationError
load_dotenv()
from langgraph.graph import StateGraph, START, END
from typing import Literal
from IPython.display import Image, display
# %%
# Different ways to define a state
# class State(TypedDict):
#     name: str
#     mood: Literal["happy", "sad"]

# @dataclass
# class DataclassState(dataclass):
#     name: str
#     mood: Literal["happy", "sad"]
    
class State(BaseModel):
    name: str
    mood: Literal["happy", "sad"]
    
    # @validator("mood")
    @field_validator("mood")
    def validate_mood(cls, value):
        # Ensure the mood is either "happy" or "sad"
        if value not in ["happy", "sad"]:
            raise ValueError("Input should be 'happy' or 'sad'")
        return value
    class Config:
        validate_assignment = True
# %%
def node1(state:State) -> State:
    print("---Node 1---")
    state.name = state.name + " is ... "
    print(state)
    # return {"name": state['name'] + " is ... "}
    return state

def node2(state:State) -> State:
    print("---Node 2---")
    state.mood = "happy"
    print(state)
    # return {"mood": "happy"}
    return state

def node3(state:State) -> State:
    print("---Node 3---")
    state.mood = "sad"
    print(state)
    # return {"mood": "sad"}
    return state

def decide_mood(state) -> Literal["node2", "node3"]:
    print("---Mood Set---")
    if random.random() > 0.5:
        return "node2"
    return "node3"

# %%
builder = StateGraph(State)
builder.add_node("node1", node1)
builder.add_node("node2", node2)
builder.add_node("node3", node3)

builder.add_edge(START, "node1")
builder.add_conditional_edges("node1", decide_mood)
builder.add_edge("node2", END)
builder.add_edge("node3", END)

graph = builder.compile()
# %%
try:
    state = State(name="Sandeep", mood="happy")
except ValidationError as e:
    raise ValueError("Each mood must be either 'happy' or 'sad'")
# %%
res = graph.invoke(state)
print(res)
# %%
