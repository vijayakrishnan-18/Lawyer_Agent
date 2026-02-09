import streamlit as st
import asyncio
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

from lawyer_graph import app_graph

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nyaya-Agent: Indian Legal Assistant", layout="wide")

st.title("⚖️ Lawyer-Agent: AI Legal Assistant")
st.markdown("""
This system uses **Agentic RAG** with **Recursive Hallucination Checks**.
1. **Retrieve:** Searches BNS/BNSS.
2. **Reason:** Uses DeepSeek-R1 for legal logic.
3. **Verify:** Uses Llama-3 to double-check citations.
""")

# --- SIDEBAR: API KEY & FILE UPLOAD ---
with st.sidebar:
    st.header("Settings & Uploads")
    api_key = st.text_input("Enter Groq API Key:", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    uploaded_file = st.file_uploader("Upload Legal Notice (PDF)", type=["pdf"])
    
    uploaded_text = ""
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            uploaded_text = "\n".join([p.page_content for p in pages])
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Describe your legal issue (e.g., 'Police stopped me at 1 AM')..."):
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- AGENT EXECUTION ---
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        
        status_placeholder.markdown("🔍 *Agents are researching BNS & BNSS...*")
        
        try:
            # Prepare inputs
            inputs = {
                "question": prompt,
                "uploaded_doc_content": uploaded_text if uploaded_text else "No specific document provided.",
                "retry_count": 0
            }
            
            # Run the LangGraph
            # We use aconfig to allow async execution inside Streamlit
            final_state = None
            for output in app_graph.stream(inputs):
                for key, value in output.items():
                    if key == "retrieve":
                        status_placeholder.markdown("📚 *Documents Retrieved. Reasoning...*")
                    elif key == "generate":
                        status_placeholder.markdown("⚖️ *Drafting Legal Advice...*")
                    elif key == "hallucination_check":
                        status_placeholder.markdown("🕵️ *Verifying citations against hallucination...*")
                final_state = value

            # Extract final response
            # Note: The stream output structure varies, usually the last state contains the answer
            final_response = final_state.get("generation", "I could not generate a response.")
            
            # If DeepSeek generated <think> tags, we can format them
            if "<think>" in final_response:
                parts = final_response.split("</think>")
                if len(parts) > 1:
                    thought_process = parts[0].replace("<think>", "").strip()
                    actual_answer = parts[1].strip()
                    
                    with st.expander("See Legal Reasoning Logic (Chain of Thought)"):
                        st.write(thought_process)
                    
                    message_placeholder.markdown(actual_answer)
                    st.session_state.messages.append({"role": "assistant", "content": actual_answer})
                else:
                    message_placeholder.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
            else:
                message_placeholder.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
            status_placeholder.empty()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Fallback for debugging
            st.write("Ensure your Groq API Key is valid and BNS documents are ingested.")