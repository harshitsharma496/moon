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

# 1. Page Setup
st.set_page_config(page_title="Local Knowledge Base", layout="wide")
st.title("🌙 Welcome to MoonGpt")

# 2. Secure API Key Handling
# We look for the key in Streamlit Secrets (Cloud)
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    # Fallback for local testing: Sidebar
    api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")

if not api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

# 3. Initialize Model
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

# 4. Define Prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a cute helpful assistant.
    First, search the following context for the answer and provide to the point short answer.
    If the answer is not found in the context, answer the question using your own general knowledge.

    <context>
    {context}
    </context>

    Question: {question}
    """
)

# 5. Load and Cache the Knowledge Base
@st.cache_resource
def load_knowledge_base():
    file_path = "knowledge.txt"

    if not os.path.exists(file_path):
        return None

    loader = TextLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

# 6. Initialize Vector Store
if "vectors" not in st.session_state:
    with st.spinner("Reading Moon Mind..."):
        vectors = load_knowledge_base()
        if vectors:
            st.session_state.vectors = vectors
            st.success("Moon Mind Loaded Successfully!")
        else:
            st.error("❌ knowledge.txt not found in repository!")
            st.stop()

# 7. Chat Interface
user_prompt = st.text_input("Ask a question with love:")

if user_prompt:
    retriever = st.session_state.vectors.as_retriever()

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    response = rag_chain.invoke(user_prompt)
    st.markdown(response)