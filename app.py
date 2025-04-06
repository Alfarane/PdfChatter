import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

#gets text from pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf.seek(0)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#divides the text from pdf to chunks            
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

#converts chunks into embeddings
def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks found. Please check your PDF upload or text extraction.")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorestore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorestore
    
def get_conversation_chain(vectorstore):
   # Load Gemini properly for LangChain
    model = ChatGoogleGenerativeAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-1.5-flash",
        temperature=0.2,  # you can set your own
        convert_system_message_to_human=True  # optional but recommended
    )

    # Setup memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Setup Conversational Retrieval Chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

    return conversation_chain

def handle_user_input(user_question):
    if "conversation" in st.session_state and st.session_state.conversation is not None:
        response = st.session_state.conversation.run({"question": user_question})
        
        # Initialize chat_history if it doesn't exist in session_state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Store the user's question and bot's response in the chat history
        st.session_state.chat_history.append({"user": user_question, "bot": response})
        
        # Display the chat history
        for chat in st.session_state.chat_history:
            st.write(user_template.replace("{{MSG}}", chat['user']), unsafe_allow_html=True)
            
            # Bot response
            st.write(bot_template.replace("{{MSG}}", chat['bot']), unsafe_allow_html=True)
    else:
        st.write("Conversation chain is not initialized. Please upload PDFs and process them.")

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_user_input(user_question)


    with st.sidebar:
        st.subheader("Your documents")

        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process", accept_multiple_files=True, type="pdf")

        if st.button("Process"):
            # User sees a spinning icon
            with st.spinner("Processing"):

                # Get the PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
            
if __name__ == '__main__':
    main()
