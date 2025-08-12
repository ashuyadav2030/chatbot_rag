import os
import re
import asyncio
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

@st.cache_resource
def load_documents():
    loader = PyPDFLoader("final_chat.pdf")
    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])

    pattern = r"(Q\d+:.*?)(?=Q\d+:|$)"
    matches = re.findall(pattern, full_text, flags=re.DOTALL)

    qa_docs = [Document(page_content=match.strip()) for match in matches]

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(qa_docs)

docs = load_documents()

@st.cache_resource
def create_vectorstore():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(docs, embeddings)

vector_store = create_vectorstore()
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

prompt = PromptTemplate(
    template="""
    You are a helpful chatbot assistant.
    Answer ONLY using the provided transcript context.
    If the context is insufficient, say "I don't know" and do not make up an answer.

    Context:
    {context}

    Question: {question}
    """,
    input_variables=["context", "question"]
)

def format_docs(docs):
    if not isinstance(docs, list):
        return ""
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    | prompt
    | model
    | StrOutputParser()
)

st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Chatbot: Ask Questions")

question = st.text_input("Ask a question based on the PDF:")

if question:
    with st.spinner("Thinking..."):
        try:
            answer = chain.invoke(question)
            st.markdown("Answer:")
            st.success(answer)
        except Exception as e:
            st.error(f"Error: {str(e)}")
