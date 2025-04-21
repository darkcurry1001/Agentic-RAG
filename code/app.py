import streamlit as st
from agent_graph import graph  # Replace with the actual file where your graph is defined

st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📚 RAG Question Answering Assistant")
st.markdown("Ask a question based on the document collection.")

# Input field
question = st.text_input("🔍 Your question", placeholder="e.g. What are the sustainability goals of L'Oréal?")

# Submit button
if st.button("Get Answer") and question:
    with st.spinner("🔎 Retrieving information and generating answer..."):
        try:
            result = graph.invoke({"question": question})
            st.success("✅ Answer generated!")

            st.markdown("### 🧠 Answer")
            st.write(result["answer"])

            # Optionally show the retrieved context documents
            with st.expander("📄 Show retrieved documents"):
                for i, doc in enumerate(result["context"]):
                    st.markdown(f"**Document {i+1}**")
                    st.text(doc.page_content[:1000])  # Trim long text

        except Exception as e:
            st.error(f"⚠️ Error: {e}")