import streamlit as st
import os
from langchain.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
import time

# Set page configuration
st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            color: #4CAF50;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .main .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<div class="title">Document Q&A Application</div>', unsafe_allow_html=True)
st.write("This application allows you to ask questions based on a set of PDF documents. The documents are loaded, split into chunks, and embedded using a Hugging Face model. These embeddings are stored in a FAISS vector store, which is used to retrieve relevant document chunks in response to user questions.")

# Sidebar for user inputs and actions
st.sidebar.title("Controls")
st.sidebar.write("Use the controls below to interact with the application.")

# Initialize session state attributes if not already initialized
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'loader' not in st.session_state:
    st.session_state.loader = None
if 'docs' not in st.session_state:
    st.session_state.docs = None
if 'text_splitter' not in st.session_state:
    st.session_state.text_splitter = None
if 'final_documents' not in st.session_state:
    st.session_state.final_documents = None

# Load the local LLaMA model
llm = Ollama(model="llama3")

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions: {input}
""")

# Function to create vector embeddings
def vector_embedding():
    if st.session_state.vectors is None:
        with st.spinner('Processing documents...'):
            # Use HuggingFaceEmbeddings for local embedding generation
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

            # Load PDFs from directory
            st.session_state.loader = PyPDFDirectoryLoader("./PDFs")
            st.session_state.docs = st.session_state.loader.load()
            st.write(f"Loaded {len(st.session_state.docs)} documents")

            # Debug: Print contents of each loaded document
            for i, doc in enumerate(st.session_state.docs):
                st.write(f"Document {i+1} - First 500 chars: {doc.page_content[:500]}...")
                if i == 2:  # Check the third document specifically
                    st.write(f"Document {i+1} Full Content: {doc.page_content}")

            # Split documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.write(f"Split into {len(st.session_state.final_documents)} chunks")

            # Debug: Print contents of the first few chunks
            for i, doc in enumerate(st.session_state.final_documents[:5]):  # Limit to first 5 chunks for brevity
                st.write(f"Chunk {i+1} - First 500 chars: {doc.page_content[:500]}...")

            # Create FAISS vector store
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.write("Vector Store DB is Ready")

prompt1 = st.sidebar.text_input("Enter Your Question From Documents")  # Sidebar input

if st.sidebar.button("Documents Embedding"):
    vector_embedding()

# Main content
if prompt1 and st.session_state.vectors is not None:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    elapsed_time = time.process_time() - start
    st.write(f"Response time: {elapsed_time:.2f} seconds")

    if 'answer' in response:
        st.markdown("### Answer")
        st.write(response['answer'])
    else:
        st.write("No answer found")

if st.session_state.vectors is not None:
    st.write(f"Vectors made in {len(st.session_state.final_documents)} chunks")
else:
    st.write("Vectors are not created yet")
