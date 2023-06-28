import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def get_PDFText(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_TextChunks(pdf_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap = 200,
        length_function = len
        )
    chunks = text_splitter.split_text(pdf_text)
    return chunks

def get_VectorStore(chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl')
    vectorStore = FAISS.from_texts(texts=chunks, embedding = embeddings)
    return vectorStore

def get_ConversationChain(vectorStore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorStore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({"question":user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title= 'MultiPDF BOT', page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")
    user_question = st.text_input("Ask any question you have about your documents: ")
    if user_question:
        handle_user_input(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #Get pdf text
                pdf_text = get_PDFText(pdf_docs)

                #Get text chunks
                chunks = get_TextChunks(pdf_text)

                #Create vector store
                vectorStore = get_VectorStore(chunks)

                #Conversation Chain
                st.session_state.conversation = get_ConversationChain(vectorStore)

if __name__ == '__main__':
    main()