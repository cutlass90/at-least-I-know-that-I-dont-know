from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()

# llm = ChatOpenAI()
llm = Ollama(model="llama3")
output_parser = StrOutputParser()
# res = llm.invoke("how can langsmith help with testing?")
# print(res)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])

chain = prompt | llm | output_parser
res = chain.stream({"input": "how can langsmith help with testing?"})
# res = chain.invoke({"input": "how can langsmith help with testing?"})
for r in res:
    print(r, end='', flush=True)
