import langchain
import streamlit as st
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

st.title("🦜️🔗 Chat with Docx File")
st.text("Upload txt file")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()
    # st.write(string_data)
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )

    documents = text_splitter.create_documents([string_data])
    st.write(documents[0])
    st.write(documents[1])
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(documents, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    query = "what did the president say about Ketanji Brown Jackson?"
    result = qa({"query": query})
    st.write(result)

prompt = st.text_area("Enter your question:")

with st.form("Form"):
    api_key = st.text_area("Enter your OpenAI key:")
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Form submitted")
