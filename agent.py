from dotenv import load_dotenv
from typing import Annotated, Dict, Any, List
from langchain.chat_models import init_chat_model

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage, HumanMessage, AnyMessage, AIMessage
from langchain_core.tools import InjectedToolCallId, tool

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PlaywrightURLLoader
from bs4 import BeautifulSoup
import json

from src.prompts import get_prompt

load_dotenv()

class State(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    execution_plan: Dict[str, Any]
    execution_results: List[Dict]
    current_step: str
    synthesis: str
    final_answer: str
    next_action: str
    error: str


@tool
def structured_web_page_extractor(url: str, selectors: str = None, timeout: int = 10, headless: bool = True) -> str:
    """
    Extracts structured data from a web page using Playwright.
    
    Parameters:
        url (str): The URL of the web page to extract data from.
        selectors (str, optional): JSON string of CSS selectors for elements to extract. Example: '{"table": ".wikitable", "list": "ul.special-list"}'
        timeout (int, optional): Max seconds to wait for page load. Default is 10.
        headless (bool, optional): Run browser in headless mode. Default is True.
    
    Returns:
        str: JSON string containing extracted data with tables, lists, and raw HTML.
    """
    try:
        # Parse selectors if provided
        selector_dict = {}
        if selectors:
            try:
                selector_dict = json.loads(selectors)
            except json.JSONDecodeError:
                selector_dict = {}
        
        # Load the page with Playwright
        loader = PlaywrightURLLoader(
            urls=[url],
            continue_on_failure=False,
            headless=headless
        )
        
        # Get the page content
        docs = loader.load()
        if not docs:
            return json.dumps({"error": "Failed to load page content"})
        
        html_content = docs[0].page_content
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract tables
        tables = []
        table_selectors = selector_dict.get('table', 'table')
        if isinstance(table_selectors, str):
            table_selectors = [table_selectors]
        
        for selector in table_selectors:
            found_tables = soup.select(selector)
            for table in found_tables:
                table_data = []
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_data = [cell.get_text(strip=True) for cell in cells]
                        table_data.append(row_data)
                if table_data:
                    tables.append(table_data)
        
        # Extract lists
        lists = []
        list_selectors = selector_dict.get('list', 'ul, ol')
        if isinstance(list_selectors, str):
            list_selectors = [list_selectors]
        
        for selector in list_selectors:
            found_lists = soup.select(selector)
            for list_elem in found_lists:
                list_items = list_elem.find_all('li')
                if list_items:
                    list_data = [item.get_text(strip=True) for item in list_items]
                    lists.append(list_data)
        
        # Extract specific elements if selectors provided
        specific_elements = {}
        for key, selector in selector_dict.items():
            if key not in ['table', 'list']:
                elements = soup.select(selector)
                specific_elements[key] = [elem.get_text(strip=True) for elem in elements]
        
        # Prepare result
        result = {
            "tables": tables,
            "lists": lists,
            "specific_elements": specific_elements,
            "url": url,
            "status": "success"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to extract data: {str(e)}", "url": url})


# Create the web search tool properly
web_search_tool = DuckDuckGoSearchResults()
web_search_tool.description = "Search the web for current information using DuckDuckGo"

# Create the web page extractor tool
web_page_extractor_tool = structured_web_page_extractor
web_page_extractor_tool.description = "Extract structured data (tables, lists) from web pages using Playwright"

# Define tools list with the proper tool instances
tools = [web_search_tool, web_page_extractor_tool]


llm_init = init_chat_model("google_genai:gemini-2.0-flash")
llm = llm_init.bind_tools(tools)


graph_builder = StateGraph(State)

# Defines nodes

# def chatbot(state: State):
#     state["question"] = state["messages"][-1].content
#     return {"messages": [llm.invoke(state["messages"])]}

def planner(state):
    question = state['question']
    prompt = get_prompt('PLANNER_PROMPT', question=question) 
    
    messages = [HumanMessage(content=prompt)]
    execution_plan_msg = llm.invoke(messages)

    execution_plan = execution_plan_msg.content
    print(f"Execution Plan: {execution_plan}")
    return {"execution_plan": execution_plan, "current_step": "Step 1"}

def executor(state):
    execution_plan = state['execution_plan']
    execution_results = state.get('execution_results', [])
    messages = state.get('messages', [])
    current_step = state.get('current_step', '')
    
    # If no current step is set, start with step 1
    if not current_step:
        current_step = "Step 1"
    
    if(len(messages) > 0):
        last_message = messages[-1]
        execution_results.append(last_message)
    
    previous_results = "\n".join(str(item) for item in execution_results)

    prompt = get_prompt('EXECUTOR_PROMPT', plan=execution_plan, previous_results=previous_results, current_step=current_step) 
    
    messages = [HumanMessage(content=prompt)]
    execution_result = llm.invoke(messages)
    
    messages = [execution_result]
    execution_results.append(execution_result)

    # Determine next step based on the plan
    # This is a simple implementation - you might want to make this more sophisticated
    # by parsing the plan to determine the actual next step
    next_step = current_step  # For now, keep the same step until verificator decides to move on
    
    print(f"Executed: {current_step}")
    print(f"Messages: {messages[-1].content}")
    return {"messages": messages, "execution_results": execution_results, "current_step": next_step}

def verificator(state):
    execution_plan = state['execution_plan']
    execution_results = state.get('execution_results', [])
    current_step = state.get('current_step', '')
    
    # Format previous results for the prompt
    previous_results = "\n".join(str(item) for item in execution_results)
    
    prompt = get_prompt('VERIFICATOR_PROMPT', 
                       plan=execution_plan, 
                       previous_results=previous_results, 
                       current_step=current_step)
    
    messages = [HumanMessage(content=prompt)]
    verification_result = llm.invoke(messages)
    
    # Extract the decision from the response (case-insensitive)
    response_content = verification_result.content.strip().lower()
    
    # Determine next action based on response
    if "synthesizer" in response_content:
        next_action = "synthesizer"
    elif "planner" in response_content:
        next_action = "planner"
    elif "executor" in response_content:
        next_action = "executor"
        # If going back to executor, advance to next step
        # Simple step advancement - you might want to make this more sophisticated
        if current_step.startswith("Step "):
            try:
                step_num = int(current_step.split(" ")[1])
                current_step = f"Step {step_num + 1}"
            except:
                current_step = "Next Step"
        else:
            current_step = "Step 2"  # Default fallback
    else:
        # Default to executor if unclear
        next_action = "executor"
    
    return {"next_action": next_action, "messages": [verification_result], "current_step": current_step}

def should_synthesize(state):
    """Determine the next node based on verificator's decision."""
    next_action = state.get('next_action', 'executor')
    
    if next_action == "synthesizer":
        return "synthesizer"
    elif next_action == "planner":
        return "planner"
    else:
        return "executor"

def synthesizer(state):
    execution_results = state.get('execution_results', [])
    question = state['question']

    prompt = get_prompt('SYNTHESIZER_PROMPT', execution_results=execution_results, question=question)

    messages = [HumanMessage(content=prompt)]
    final_answer_msg = llm.invoke(messages)

    final_answer = final_answer_msg.content
    return {"final_answer": final_answer}

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_node("planner", planner)
graph_builder.add_node("executor", executor)
graph_builder.add_node("verificator", verificator)
graph_builder.add_node("synthesizer", synthesizer)


graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "executor")
graph_builder.add_conditional_edges("executor", tools_condition)
graph_builder.add_edge("executor", "verificator")
graph_builder.add_edge("tools", "verificator")
graph_builder.add_conditional_edges(
    "verificator",
    should_synthesize,
    {
        "synthesizer": "synthesizer",
        "planner": "planner", 
        "executor": "executor"
    }
)
graph_builder.add_edge("synthesizer", END)

graph = graph_builder.compile()

def main():
    result = graph.invoke({"question" : "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."})
    final_answer = result["final_answer"]
    print(final_answer)

if __name__ == "__main__":
    main() 

# Handles user input
# def stream_graph_updates(user_input: str):
#     for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)


# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break
#         stream_graph_updates(user_input)
#     except:
#         # fallback if input() is not available
#         user_input = "What do you know about LangGraph?"
#         print("User: " + user_input)
#         stream_graph_updates(user_input)
#         break

