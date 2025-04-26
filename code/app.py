import streamlit as st
from agent_graph import graph, invoke_graph
from agent_graph import chat_history

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("💬 RAG Chat Assistant")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = chat_history
if "new_user_input" not in st.session_state:
    st.session_state.new_user_input = None
if "input_submitted" not in st.session_state:
    st.session_state.input_submitted = False

# Sidebar for toggling chat history
with st.sidebar:
    st.header("📜 Chat History")
    if st.toggle("Show full chat history", value=False):
        st.text_area("History content", st.session_state.chat_history, height=300)

# Display previous messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Handle user input
user_question = st.chat_input("Ask your question here...")

# If the user submits a question
if user_question and not st.session_state.input_submitted:
    st.session_state.new_user_input = user_question
    st.session_state.input_submitted = True

# Now process the new user input
if st.session_state.input_submitted:
    question = st.session_state.new_user_input

    # Display user message
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        try:
            result = invoke_graph(
                graph,
                question=question,
                chat_history=st.session_state.chat_history
            )

            answer = result["answer"]

            # Display assistant response
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Update the chat history
            st.session_state.chat_history = result["chat_history"]

        except Exception as e:
            st.error(f"Error: {e}")

    # After handling, reset the submission flag
    st.session_state.input_submitted = False