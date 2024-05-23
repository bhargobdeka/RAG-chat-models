from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from fastapi import FastAPI
from langserve import add_routes
from langchain.pydantic_v1 import BaseModel
from langchain_core.output_parsers import StrOutputParser


# Setting environment variables and setup tracking
import os
from dotenv import load_dotenv
## access the openai api key from .env file and map it to the os system.
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
## Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY") # where monitoring results needs to be stored

# Loading document
loader = PyPDFLoader("../documents/budget-2024.pdf")
document = loader.load()

# Split text -- default one
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
text = text_splitter.split_documents(document)

# Embedding
embeddings_model = OpenAIEmbeddings()

# Chroma vectorstore
db_chroma = Chroma.from_documents(document, embeddings_model)

# Retriever
retriever = db_chroma.as_retriever(search_type="mmr")

# Load the llm 
llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)

##Creating Chain ##

# Prompt Template
prompt = ChatPromptTemplate.from_template("""You are an expert on Canada's Budget. Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Create a chain 
doc_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, doc_chain)


##. Creating the fastAPI app

# App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server to interact with Canada Budget 2024.",
)

# 5. Adding chain route


add_routes(
    app,
    chain,
    path="/budget_chain"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)