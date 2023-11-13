import os

from dotenv import load_dotenv
import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
FAISS_DB_DIR = os.environ["FAISS_DB_DIR"]

MODEL_NAME = "gpt-3.5-turbo-16k-0613"
MODEL_TEMPERATURE = 0.9

st.title("Hakky ChatBot")

# メッセージ履歴を保持するリストの定義
if "messages" not in st.session_state:
    st.session_state.messages = []

# メッセージ履歴の表示
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Hakkyについて知りたいことはありますか？"):

    # ユーザーによる質問の保存・表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    model = ChatOpenAI(model=MODEL_NAME, temperature=MODEL_TEMPERATURE, client=openai.ChatCompletion)
    faiss_db = FAISS.load_local(FAISS_DB_DIR, embeddings=OpenAIEmbeddings(client=openai.ChatCompletion))

    # LLMによる回答の生成
    qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=faiss_db.as_retriever())
    query = f"あなたはHakkyについての質問に答えるChatBotです。次の質問に答えてください。:{prompt}"
    res = qa.run(query)

    # LLMによる回答の表示
    with st.chat_message("assistant"):
        st.markdown(res)

    # LLMによる回答の保存
    st.session_state.messages.append({"role": "assistant", "content": res})
