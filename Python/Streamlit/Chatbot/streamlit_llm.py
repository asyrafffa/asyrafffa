import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# App config
st.set_page_config(page_title="Chatbot", page_icon="👾")
st.title("Chatbot")

def get_response(user_query, chat_history):

    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Using LM Studio Local Inference Server
    llm = ChatOpenAI(base_url="http://localhost:1234/v1")

    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
    
# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))
    
#Clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    
# Pass preferences to get_response function
if "preferences" not in st.session_state:
    st.session_state.preferences = {"temperature": 0.7, "model": "bartowski/Gemma-2-9B-It-SPPO-Iter3-GGUF"}

st.sidebar.title("Preferences")
st.session_state.preferences["temperature"] = st.sidebar.slider("Temperature", 0.0, 1.0, st.session_state.preferences["temperature"])
st.session_state.preferences["model"] = st.sidebar.selectbox("Model", ["bartowski/Gemma-2-9B-It-SPPO-Iter3-GGUF", "other_model"])

# Modify the template to include language preference
if "language" not in st.session_state:
    st.session_state.language = "en"

# Choose language
st.sidebar.title("Language")
st.session_state.language = st.sidebar.selectbox("Select Language", ["en", "es", "fr", "de"])

