# %%
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pprint import pprint
from langchain_core.messages import SystemMessage,HumanMessage,RemoveMessage
load_dotenv()
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
# %%
class State(MessagesState):
    summary: str
    
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Define the logic to call the model
def call_model(state: State):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    return {"messages": model.invoke(messages)}

# %%
# Determine whether to end or summarize the conversation
def should_continue(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

# %%
def summarize_conversation(state: State):
    
    summary = state.get("summary", "")
    
    if summary:
        summary_msg = (f"This is the summary of the conversation upto date: {summary}\n\n"
                       "Extend the Conversation summary with the above messages.")    
    else:
        summary_msg = f"Summarize the conversation of the above messages."
        
    messages = state['messages'] + [HumanMessage(content=summary_msg)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-2]]
    return {'summary': response.content, 'messages': delete_messages}

# %%
memory = MemorySaver()
workflow = StateGraph(State)
workflow.add_node("call_model", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", should_continue)
workflow.add_edge("summarize_conversation", END)
graph = workflow.compile(checkpointer=memory)

# %%
config = {"configurable": {"thread_id": "1"}}
final_response = graph.invoke({"messages": [HumanMessage(content="What is the capital of France?")]},config=config)
print(final_response)
# %%
final_response = graph.invoke({"messages": [HumanMessage(content="Can you tell me more about it")]},config=config)
print(final_response)

# %%
