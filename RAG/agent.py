import os
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent, create_json_agent
from langchain.agents import AgentExecutor
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

embeddings = OllamaEmbeddings(model="llama3")
text_splitter = RecursiveCharacterTextSplitter()

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

search = TavilySearchResults()
tools = [retriever_tool, search]

prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
question = "how can langsmith help with testing?"
print(question)
res = agent_executor.invoke({"input": question})
print(res['output'])
print('\n\n')

question = "what do you know about Nazar Shmatko?"
print(question)
res = agent_executor.invoke({"input": question})
print(res['output'])
