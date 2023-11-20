import os
import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.vectorstores import FAISS
from langchain.tools.base import BaseTool
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.embeddings.openai import OpenAIEmbeddings


load_dotenv()

#openai_api_key=os.environ["OPENAI_API_KEY"]

# エージェントチェーンの作成
def create_agent_chain():
    ### FAISS vectorのロード
    vectoreStore = FAISS.load_local("faiss_index/", OpenAIEmbeddings())
    ## Retriever
    retriever = vectoreStore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    ### プロンプト(Q&A)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k"), chain_type="stuff", retriever=retriever, return_source_documents=False)


if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()


st.title("langchain-streamlit-app")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
