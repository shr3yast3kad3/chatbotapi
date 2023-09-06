orig = r'''PREFIX = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed of you:"""'''

replacement = r'''PREFIX = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed of you:

Summary of the whole conversation:
{chat_history_summary}

Last few messages between you and user:
{chat_history_buffer}

Entities that the conversation is about:
{chat_history_KG}
"""'''

print(os.listdir(r'/opt/render/project/src/.venv/lib/python3.10/site-packages/langchain/agents/agent_toolkits/pandas'))

file_loc = r'/opt/render/project/src/.venv/lib/python3.10/site-packages/langchain/agents/agent_toolkits/pandas/prompt.py'

with open(file_loc, 'r') as file:
    data = file.read()
    data = data.replace(orig, replacement)
  
with open(file_loc, 'w') as file:
    file.write(data)

#importing dependencies

import json
from fastapi import FastAPI
import os
import pandas as pd
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langcorn import create_service
import spacy
from langchain.memory import ConversationEntityMemory
from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.memory import ConversationKGMemory
from langchain.memory import CombinedMemory

#with open('keys.json') as json_file:
#    api_key = json.load(json_file)['api_key']
#os.environ["OPENAI_API_KEY"] = api_key

with open(r'prompt.py', 'r') as file:
    data = file.read()
    data = data.replace(orig, replacement)

with open(r'prompt.py', 'w') as file:
    file.write(data)

print("Text Replaced")


document = pd.read_csv('./data/train.csv')
document.rename(columns = {'Lead_Creation_Date':'Date'}, inplace = True)
document['Date'] = pd.to_datetime(document['Date'], format="%d/%m/%y")
document['DOB'] = pd.to_datetime(document['DOB'], format="%d/%m/%y")

llm_code = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k") #gpt-3.5-turbo-16k-0613
llm_context = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo-16k") #gpt-3.5-turbo

chat_history_buffer = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history_buffer",
    input_key="input"
    )

chat_history_summary = ConversationSummaryMemory(
    llm=llm_context, 
    memory_key="chat_history_summary",
    input_key="input"
    )

chat_history_KG = ConversationKGMemory(
    llm=llm_context, 
    memory_key="chat_history_KG",
    input_key="input",
    )

memory = CombinedMemory(memories=[chat_history_buffer, chat_history_summary, chat_history_KG])

little_guy_with_memory = create_pandas_dataframe_agent(
    llm = llm_code, 
    df = document, 
    verbose=True, 
    agent_executor_kwargs={"memory": memory},
    input_variables=['df_head', 'input', 'agent_scratchpad', 'chat_history_buffer', 'chat_history_summary', 'chat_history_KG']
    )

app = FastAPI()

@app.get('/')
async def root(prompt: str):
	return little_guy_with_memory.run("Hi!")

@app.get('/little_guy/{prompt}')
async def root(prompt: str):
	return little_guy_with_memory.run(prompt + "Use python_repl_ast.")
