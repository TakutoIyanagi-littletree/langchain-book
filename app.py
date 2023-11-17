from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
    )

from langchain.vectorstores import pinecone

def doc_preprocessing() :
    loader = DirectoryLoader(
        'data/',
        glob = '**/*.pdf', 
        show_progress =True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split
        
        
@st.cache_resource
def embedding_db():
    embeddings = OpenAIEmbeddings()
    docs_split = doc_preprocessing()
    doc_db = pinecone.from_documents(
        docs_split,
        embeddings,
        index_name='st-app'
    )
    return doc_db
    
llm = ChatOpenAI()
doc_db = embedding_db()

def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type='stuff',
        retriever = doc_db.as_retriever(),
        query = query,
    )
    result = qa.run(query)
    return result
    
def main():
    st.tittle("Question and Answering App powerd by LLM and pinecone")
    text_input = st.text_input("Ask your query")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your query:" + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)

if __name__ == "__maln__":
    main()
