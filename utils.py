from langchain.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import re

def format_response(response: str) -> str:
    # Clean and format the response for better readability
    formatted_response = re.sub(r'\s+', ' ', response).strip()  # Replace multiple spaces/newlines with a single space
    formatted_response = formatted_response.replace("\n", "<br>")  # Replace newlines with HTML line breaks
    formatted_response = formatted_response.replace("Human:", "<strong>Human:</strong>")  # Make "Human:" bold
    formatted_response = formatted_response.replace("context =", "<strong>Context:</strong>")  # Make "context:" bold
    formatted_response = formatted_response.replace("question =", "<strong>Question:</strong>")  # Make "question:" bold

    return formatted_response

def load_knowledgeBase():
    print('initializing knowledge base...')
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db = FAISS.load_local(DB_FAISS_PATH, 
                          embeddings, 
                          allow_dangerous_deserialization=True)

    
    print('Knowledge base initialized!')
    return db

def load_prompt():
    prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
    Given below is the context and question of the user.
    context = {context}
    question = {question}
    if the answer is not in the pdf answer "i do not know what the hell you are asking about"
        """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_llm():
    print("initializing llm....")
    llm = HuggingFacePipeline.from_model_id(model_id="gpt2",
                                            task="text-generation", 
                                            pipeline_kwargs={
                                                "max_new_tokens": 400,
                                                "top_p": 0.95, 
                                                "do_sample": True,
                                                "top_k": 50,
                                                "temperature": 0.2,
                                                "repetition_penalty": 2.0})
    
    print("llm initialized!")

    return llm