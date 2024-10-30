import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
import os 
import torch




def get_pdf_text(pdfs) :
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
    

def get_vectorstore(text_chunks):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding = embeddings)
    return vectorstore  

def get_conversation_chain(vector_store):
  
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response)

def main():
    load_dotenv()

    st.set_page_config(page_title="chatWme", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    st.header("chatWme")
    user_question = st.text_input("Ask a question")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "hello robot"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your Documents")
        pdfs = st.file_uploader("Upload your documents", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdfs)
            
                #get the text chunks
                text_chunks = get_text_chunks(raw_text)

                #create vector store
                vector_store = get_vectorstore(text_chunks)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

    
    


if __name__ == '__main__' :
    main()
