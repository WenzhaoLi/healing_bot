import streamlit as st
from st_chat_message import message
import pandas as pd
from datetime import datetime
from heal_agent import heal_agent

st.title("Healing Chat Bot :heart:")
tab1, tab2 = st.tabs(["Chat", "Usage Chart"])

if "llm_chain" not in st.session_state:
    st.session_state.llm_chain = heal_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

def append_state_messages(user_message, bot_message):
    st.session_state.messages.append({"user_message": user_message, "bot_message": bot_message})

def restore_history_messages():
    for history_message in st.session_state.messages:
        message(history_message["user_message"], is_user=True, key=str(datetime.now()))
        message(history_message["bot_message"], is_user=False,key=str(datetime.now()))

user_message = st.chat_input(placeholder="Type a message...")
with tab1:
    st.header("Chat")
    if user_message:
        restore_history_messages()
        message(user_message, is_user=True, key="user_message")
        output = st.session_state.llm_chain.generate(query=user_message)
        message(output, is_user=False, key="bot_message")
        append_state_messages(user_message, output)


with tab2:
    st.header("OpenAI Usage Chart")
    df = pd.read_csv("openai_api_usage.csv")
    st.bar_chart(df, x="us_date_format", y="total_cost_usd")

    
