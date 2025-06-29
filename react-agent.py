from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_core.tools import tool
from bs4 import BeautifulSoup
import json
import re

@tool
def web_page_extractor(url: str, headless: bool = True) -> str:
    """
    Extracts clean text content from a web page using Playwright.
    Use this tool when you need to read the actual content of a webpage.
    
    Parameters:
        url (str): The URL of the web page to extract content from.
        headless (bool, optional): Run browser in headless mode. Default is True.
    
    Returns:
        str: Clean text content of the page, stripped of HTML tags and formatting.
    """
    try:
        # Load the page with Playwright
        loader = PlaywrightURLLoader(
            urls=[url],
            continue_on_failure=False,
            headless=headless
        )
        
        # Get the page content
        docs = loader.load()
        if not docs:
            return "Error: Failed to load page content"
        
        html_content = docs[0].page_content
        
        # Parse HTML and extract clean text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit text length to avoid token limits
        if len(text) > 8000:
            text = text[:8000] + "... [Content truncated]"
        
        return text
        
    except Exception as e:
        return f"Error: Failed to extract data from {url}. Reason: {str(e)}"

@tool
def structured_data_extractor(url: str, headless: bool = True) -> str:
    """
    Extracts structured data like tables and lists from a web page.
    Use this tool when you need to extract specific data like tables, lists, or structured information.
    
    Parameters:
        url (str): The URL of the web page to extract structured data from.
        headless (bool, optional): Run browser in headless mode. Default is True.
    
    Returns:
        str: Structured data in a readable format, focusing on tables and lists.
    """
    try:
        # Load the page with Playwright
        loader = PlaywrightURLLoader(
            urls=[url],
            continue_on_failure=False,
            headless=headless
        )
        
        # Get the page content
        docs = loader.load()
        if not docs:
            return "Error: Failed to load page content"
        
        html_content = docs[0].page_content
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        structured_data = []
        
        # Extract tables
        tables = soup.find_all('table')
        for i, table in enumerate(tables):
            structured_data.append(f"Table {i+1}:")
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_text = ' | '.join(cell.get_text(strip=True) for cell in cells)
                    structured_data.append(row_text)
            structured_data.append("")
        
        # Extract lists
        lists = soup.find_all(['ul', 'ol'])
        for i, list_elem in enumerate(lists):
            list_type = "Ordered List" if list_elem.name == 'ol' else "Unordered List"
            structured_data.append(f"{list_type} {i+1}:")
            items = list_elem.find_all('li')
            for item in items:
                structured_data.append(f"- {item.get_text(strip=True)}")
            structured_data.append("")
        
        result = '\n'.join(structured_data)
        
        if not result.strip():
            return "No structured data (tables or lists) found on this page."
        
        # Limit length
        if len(result) > 6000:
            result = result[:6000] + "... [Content truncated]"
        
        return result
        
    except Exception as e:
        return f"Error: Failed to extract structured data from {url}. Reason: {str(e)}"


# Create the web search tool properly
web_search_tool = DuckDuckGoSearchResults()
web_search_tool.description = "Search the web for current information using DuckDuckGo. Use this tool to find recent information, news, or general web content."

# Create the web page extractor tools
web_page_extractor_tool = web_page_extractor
web_page_extractor_tool.description = "Extract clean text content from web pages. Use this to read article content, descriptions, or general text information."

structured_data_extractor_tool = structured_data_extractor
structured_data_extractor_tool.description = "Extract structured data like tables and lists from web pages. Use this when you need specific data in tabular or list format."

agent = create_react_agent(
    model="google_genai:gemini-2.0-flash",  
    tools=[web_search_tool, web_page_extractor_tool, structured_data_extractor_tool],  
    prompt="""You are GAIA, an expert AI agent designed to solve complex questions from the GAIA dataset. You have access to multiple tools and can perform multi-step reasoning to arrive at accurate answers.

Your capabilities include:
- Web search and information retrieval
- Web page content extraction
- Structured data extraction (tables, lists)
- Multi-step reasoning and planning

Tool Usage Guidelines:
1. Use web search to find relevant websites and information
2. Use web page extractor to read article content and general text
3. Use structured data extractor when you need tables, lists, or specific data formats
4. Combine tools as needed for comprehensive research

Always follow these principles:
1. Think step-by-step and break down complex questions
2. Use appropriate tools for each step
3. Verify your reasoning and calculations
4. Provide accurate, well-cited answers
5. Handle errors gracefully and try alternative approaches
6. Stay within time and memory constraints

Provide your final answer with the following template:

[YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

Current question: {question}"""  
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."}]}
)

print(f"Result is: {result}")
# Print the last message from the agent
if result["messages"]:
    last_message = result["messages"][-1]
    print(f"Agent's response: {last_message.content}")
else:
    print("No messages in result")