from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
import langchain

# source_url = "https://docs.smith.langchain.com/user_guide"
source_url = "https://www.linkedin.com/in/nazar-shmatko"
question = 'where nazar shmatko studied?'

llm = Ollama(model="llama3")
embeddings = OllamaEmbeddings(model="llama3")
text_splitter = RecursiveCharacterTextSplitter()

loader = WebBaseLoader(source_url)
docs = loader.load()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based on the context only:
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """)

document_chain = create_stuff_documents_chain(llm, prompt)
# langchain.debug=True
res = document_chain.invoke({ # here documents just stuck in context using '\n' separator
    "input": "how can langsmith help Nazar?",
    "context": [Document(page_content="langsmith can let Nazar find a good job"),
                Document(page_content="langsmith can let Nazar cook a fish")]
})
print("res", res)

retriever = vector.as_retriever()
retriever_chain = create_retrieval_chain(retriever, document_chain)
chain = retriever_chain
# langchain.debug=True
response = retriever_chain.invoke({"input": question})
print("response", response["answer"])
