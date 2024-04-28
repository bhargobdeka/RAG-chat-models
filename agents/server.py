import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import create_tool_calling_agent, initialize_agent, load_tools
from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool

from fastapi import FastAPI
from langserve import add_routes
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from typing import List
import uvicorn
load_dotenv()

## openai api
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

## google api key
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY") # note: the api key should be doubles quotes. i.e., ""
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") # google search api
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID") # google custom search engine id
## Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") # where monitoring results needs to be stored

## tavily api
os.environ['TAVILY_API_KEY']= os.getenv("TAVILY_API_KEY")

# search tool using Tavily
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search)

# search tool from google
google_tool = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper())

search = GoogleSearchAPIWrapper()
google_search_tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

# yahoo search tool
yahoo_tool = YahooFinanceNewsTool()
## Prompt

instructions = """You are an Stock Trading Expert. 
Consider the folllowing metrics to scrren the different stocks:
1. Small CAP company
2. Price-to-Earnings Ratio to be positive but lower than industry average.
3. Earnings Growth Potential of 20 percent or more in the coming years.
4. A debt-to-equity ratio of 0.5.
5. Return on Equity (ROE) of around 20-30 percent.
6. Price-to-Sales Ratio is lower than the industry average.
"""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

## OpenAI LLM
# llm = OpenAI()
# llm =ChatOpenAI(temperature=0)
# llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

## Tools
tools=[tavily_tool, google_tool, yahoo_tool, google_search_tool] 

## add more tools here

## creating agent
agent = create_openai_functions_agent(llm,tools,prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)


class Input(BaseModel):
    input: str
    

class Output(BaseModel):
    output: str

## setting up fastapi
app = FastAPI(
    title='Search for Stocks',
    version="1.0",
    decsription="A simple API Server")

## add routes
add_routes(app, 
           agent_executor.with_types(input_type=Input, output_type=Output), 
           path='/stock-trader')

if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)