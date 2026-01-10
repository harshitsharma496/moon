import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Chat with your Text File (Groq)", layout="wide")
st.title("⚡ Chat with your .txt file using Groq")

# Sidebar for API Key


# Initialize LLM
llm = ChatGroq(groq_api_key="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",
               model_name="llama-3.1-8b-instant" #
               )

# Define Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file is not None:
    # Save temp file
    with open("temp.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        loader = TextLoader("temp.txt")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.success("File processed!")

    user_prompt = st.text_input("Ask a question about your file:")

    if user_prompt:
        retriever = st.session_state.vectors.as_retriever()

        # --- THE NEW WAY (LCEL) ---
        # This replaces create_retrieval_chain and create_stuff_documents_chain
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        # Get Response
        response = rag_chain.invoke(user_prompt)
        st.write(response)

    # Cleanup
    if os.path.exists("temp.txt"):
        os.remove("temp.txt")