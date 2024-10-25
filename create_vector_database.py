from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def create_knowledgeBase(path_to_db: str):
    DB_FAISS_PATH = path_to_db
    loader = PyPDFLoader("./ischaemic_stroke_review_donnan_2019.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    vectorstore.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    path_to_db = 'vectorstore/db_faiss'
    create_knowledgeBase()