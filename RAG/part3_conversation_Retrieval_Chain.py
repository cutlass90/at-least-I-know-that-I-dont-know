from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

source_url = "https://www.linkedin.com/in/nazar-shmatko"
# question = 'where nazar shmatko studied?'
load_dotenv()

llm = Ollama(model="llama3")
embeddings = OllamaEmbeddings(model="llama3")
text_splitter = RecursiveCharacterTextSplitter()
loader = WebBaseLoader(source_url)

docs = loader.load()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

# chat_history = [
#     HumanMessage(content="Can LangSmith help test my LLM applications?"),
#     # HumanMessage(content="Can zebra eat lion?"),
#     AIMessage(content="yes")
# ]
# inputs = {
#         "chat_history": chat_history,
#         "input": "Tell me how"}
# nazar_chain = prompt | llm
# res_nazar = nazar_chain.invoke(inputs)
# print(res_nazar)

# res = retriever_chain.invoke(inputs)
# print(res)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [SystemMessage(content="You are helpful assistant that answer questions clearly and succinctly")]
while True:
    print('enter your question below')
    user_message = input()
    res2 = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": user_message
    })
    print(f"Assistant: {res2['answer']}")
    chat_history = res2['chat_history'] + [HumanMessage(content=user_message), AIMessage(content=res2['answer'])]
print()


