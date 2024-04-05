import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import google.generativeai as genai
from langchain.chat_models import ChatOpenAI


from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from htmltemplate import css, bot_template, user_template
from langchain_community.document_loaders import TextLoader

from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(docs):


                
    text = ""
    for file in docs:
        bytes_data = file.read()
        file_name = os.path.join('./',file.name)

        with open(file_name, 'wb') as f:
            f.write(bytes_data)
        
        name, extension = os.path.splitext(file_name)
        
        if extension == '.pdf':
            from langchain.document_loaders import PyPDFLoader
            print(f'Loading {file}')
            loader = PyPDFLoader(file_name)
        elif extension == '.docx':
            from langchain.document_loaders import Docx2txtLoader
            print(f'Loading {file}')
            loader = Docx2txtLoader(file_name)
        elif extension == '.txt':
            from langchain.document_loaders import TextLoader
            loader = TextLoader(file_name)
        else:
            print('Document format is not supported!')
            return None
        
        data = loader.load()
        for page in data:
            text += page.page_content
        os.remove(file_name)
    return text




    # text = ""
    # for doc in docs:
    #     pdf_reader = PdfReader(pdf)
    #     for page in pdf_reader.pages:
    #         text+=page.extract_text()
    # return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators= "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    # vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm =llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt = prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization =True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title = "Chat PDF", page_icon=":books:" )
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs💁")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        handle_user_input(user_question)

    # st.write(user_template.replace("{{MSG}}", "Hello Robert"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)

    # if user_question:
    #     user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vector_store(text_chunks)
                
                st.session_state.conversation = get_conversational_chain(vectorstore)
                
                st.success("Done")



if __name__ == "__main__":
    main()