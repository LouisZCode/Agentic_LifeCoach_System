"""
Here you will find the retriever tool, so our Assistant Agent is able to read the vector store

You will find here: 
retriever_tool
"""

from langchain_core.tools import  create_retriever_tool
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL, VECTOR_STORE_PATH
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs={"k" : 5}
)

retriever_tool = create_retriever_tool(
    retriever,
    name="retriever_tool",
    description="this tool gives you information about life coaching strategies, methods and techniques"
)