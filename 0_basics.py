# %%
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage # Core contains pillars of langchain or langgraph
from langchain_community.tools.tavily_search import TavilySearchResults # Community contains tools
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
# %%
load_dotenv()
chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# %%
resp = chat_llm.invoke("Hello, how are you?")
# %%
resp.response_metadata
# %%
tavily_search = TavilySearchResults(max_results=3)
search_docs = tavily_search.invoke("What is LangGraph?")
# %%
print(resp.content)
# %%
