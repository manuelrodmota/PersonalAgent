# import os
# from typing import Annotated
# from langchain.chat_models import init_chat_model

# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph
# from langgraph.graph.message import add_messages

# os.environ["GOOGLE_API_KEY"] = ""
# llm = init_chat_model("google_genai:gemini-2.0-flash")

# class State(TypedDict):
#     messages: Annotated[list, add_messages]
#     name: str
#     birthday: str

# graph_builder = StateGraph(State)

# def create_model(tools: list):
#     llm_with_tools = llm.bind_tools(tools)
#     return llm_with_tools







