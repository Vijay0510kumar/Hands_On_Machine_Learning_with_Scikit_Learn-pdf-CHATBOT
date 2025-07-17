import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq


# ------------------ Load .env only once ------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ------------------ Cache Heavy Resources ------------------
@st.cache_resource
def load_vector_store():
    loader = PyPDFLoader('Hands_On_Machine_Learning_with_Scikit_Le.pdf')
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


@st.cache_resource
def load_llm():
    return ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
        groq_api_key=GROQ_API_KEY
    )


# ------------------ Load Once ------------------
vector_store = load_vector_store()
llm = load_llm()


# ------------------ Chains ------------------
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step-by-step before providing a detailed answer.
If you don't know the answer, simply say "I don't know."

<context>
{context}
</context>

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="PDF Chatbot 📄", page_icon="📄")
st.title("📄 PDF Chatbot (Hands-On ML Book)")
st.write("Ask a question based on **Hands-On Machine Learning with Scikit-Learn** (Preloaded PDF).")

user_question = st.text_input("Ask your question:")

if user_question:
    with st.spinner("Thinking..."):
        response = retrieval_chain.invoke({"input": user_question})
        answer = response.get("answer", "No answer found.")
        st.markdown("### 📑 Answer:")
        st.markdown(f"> {answer}")
else:
    st.info("Please type a question to get started.")
