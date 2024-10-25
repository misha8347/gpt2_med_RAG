from fastapi import FastAPI, Query
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel

#local imports
from utils import load_knowledgeBase, load_llm, load_prompt, format_docs, format_response

class QueryModel(BaseModel):
    query: str

knowledge_base = load_knowledgeBase()
llm = load_llm()
prompt = load_prompt()
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

input = "What is the ischemic stroke?"
similar_embeddings=knowledge_base.similarity_search(input)
similar_embeddings=FAISS.from_documents(documents=similar_embeddings, 
                                        embedding=embeddings)

retriever = similar_embeddings.as_retriever()
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'Hello World!'} 

@app.post('/generate_response')
async def generate_response(data: QueryModel):
    response = rag_chain.invoke(data.query)
    formatted_response = format_response(response)

    return {'response': formatted_response}

@app.post('/generate_simple_response')
async def generate_simple_response(data: QueryModel):
    response = llm(data.query)
    formatted_response = format_response(response)

    return {'response': formatted_response}