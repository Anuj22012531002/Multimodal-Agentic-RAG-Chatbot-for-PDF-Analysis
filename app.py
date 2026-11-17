import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from tools import process_and_index_pdfs
from agents import create_multimodal_rag_graph

st.set_page_config(page_title="Multimodal Agentic AI", page_icon="🤖", layout="wide")
st.title("🤖 Multimodal Agentic AI")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False


with st.sidebar:
    st.header("Configuration")

    provider = st.selectbox("Select LLM Provider", ["Google_Gemini"])
    api_key = st.text_input(f"Enter {provider} API Key", type="password")

    st.divider()


    if st.button("Start New Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.header(" Document Upload ")
    uploaded_files = st.file_uploader(
        "Uploade one or more PDFs (containing text,image,tables)",
        type=["pdf"],
        accept_multiple_files=True,

    )

    if st.button("Process and Indexed PDFs" ):
        if not api_key:
            st.warning(f"Please enter the API Key for {provider} to proceed")
        elif not uploaded_files:
            st.warning("Please upload one or more PDF files to proceed")
        else:
            with st.spinner("Processing and indexing PDFs..."):
                try:
                    status_msg = process_and_index_pdfs(uploaded_files, provider, api_key)
                    st.session_state.indexed = True
                    st.success(status_msg)
                except Exception as e:
                    st.error(f"Error processing PDFs: {e}")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)



user_query = st.chat_input("Ask a question about the uploaded PDFs")
if user_query:
    if not api_key:
        st.error("Please enter the API Key for the selected LLM provider to proceed.")
        st.stop()

    if not st.session_state.indexed:
        st.warning("Please process and index the PDFs before asking questions.")
        st.stop()


    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                graph = create_multimodal_rag_graph(provider, api_key)
                response_state = graph.invoke({"messages": st.session_state.messages})
                final_response = response_state["messages"][-1].content[0]["text"]
                
                st.markdown(final_response)

                st.session_state.messages.append(AIMessage(content=final_response))
            except Exception as e:
                st.error(f"Error generating response: {e}")