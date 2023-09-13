import langchain
import streamlit as st
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import os


def read_file_as_string(file):
    bytes_data = file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()
    return string_data


def create_documents(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )

    documents = text_splitter.create_documents([text])
    return documents


def create_embeddings():
    embeddings = OpenAIEmbeddings()
    return embeddings


def create_vector_store(documents, embeddings):
    db = Chroma.from_documents(documents, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    return retriever


def answer_question(query, retriever):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff",
        retriever=retriever, return_source_documents=True)

    result = qa({"query": query})

    return result



st.title("🦜️🔗 Chat with Docx File")
st.text("Upload txt file")
uploaded_file = st.file_uploader("Choose a file")

prompt = st.text_area("Enter your question:")

with st.form("Form"):
    api_key = st.text_area("Enter your OpenAI key:")

    submitted = st.form_submit_button("Submit")
    if submitted:

        os.environ["OPENAI_API_KEY"] = api_key
        if uploaded_file:
            if prompt:
                text = read_file_as_string(uploaded_file)
                documents = create_documents(text)
                embeddings = create_embeddings()
                retriever = create_vector_store(documents, embeddings)
                answer = answer_question(prompt, retriever)
                st.write(answer)
