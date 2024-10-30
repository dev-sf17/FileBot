import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import time
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader

load_dotenv()
os.environ['GROQ_API_KEY']=st.secrets["GROQ_API_KEY"]

## Set up streamlit
st.set_page_config(page_title="LangChain: Your CSV and PDF Agent",page_icon="🦜")
st.title("🦜 LangChain: CSV and PDF")

llm=ChatGroq(model_name="mixtral-8x7b-32768")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def vector_embedding(docs):
    if "vectors" not in st.session_state:
        # Initialize the embeddings model
        st.session_state.embeddings = embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        st.session_state.text_splitter = text_splitter
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)
        st.session_state.vectors = Chroma.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    return st.session_state.vectors.as_retriever()

def pdf_processing(files):
        documents=[]
        for uploaded_file in files:
            temp_pdf=f"./temp.pdf"
            with open(temp_pdf,'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temp_pdf)
            docs=loader.load()
            documents.extend(docs)
        return documents    

def csv_processing(files):
    documents = []        
    
    for uploaded_file in files:
        try:
            # Save uploaded file temporarily
            temp_csv = "./temp.csv"
            with open(temp_csv, 'wb') as file:
                file.write(uploaded_file.getvalue())

            # Load the CSV file
            loader = CSVLoader(temp_csv, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
        
        except Exception as e:
            print(f"Error processing file {uploaded_file.name}: {e}")
    
    return documents
def prompt_retrieval(documents,prompt_question,llm,prompt):
    # Split and create embeddings for the documents

        if prompt_question :
            document_chain=create_stuff_documents_chain(llm,prompt)
            retriever=vector_embedding(documents)
            retrieval_chain=create_retrieval_chain(retriever,document_chain)
            start=time.process_time()
            response=retrieval_chain.invoke({'input':prompt_question})
            print("Response time :",time.process_time()-start)
            st.write(response['answer'])

            # With a steamlit expander
            with st.expander("Document Similarity Search"):
                #Find relevant chunks
                for i,doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("----------------------------")


prompt=ChatPromptTemplate.from_template(
    """ 
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}

   """
)

uploaded_files_pdf=st.file_uploader("Choose a PDF File",type="pdf",accept_multiple_files=True)
uploaded_files_csv=st.file_uploader("Choose a CSV File",type="csv",accept_multiple_files=True)
prompt_question=st.text_input("Enter your Question from Documents")
if st.button("Submit", key="submit_button"):
    if uploaded_files_csv and not uploaded_files_pdf and prompt_question:
            st.write("Uploaded CSVs")
            documents_csv=csv_processing(uploaded_files_csv)
            prompt_retrieval(documents_csv,prompt_question,llm,prompt)
    elif uploaded_files_pdf and not uploaded_files_csv and prompt_question:
            st.write("Uploaded PDFs")
            documents_pdf=pdf_processing(uploaded_files_pdf)
            prompt_retrieval(documents_pdf,prompt_question,llm,prompt)     
    elif uploaded_files_csv and uploaded_files_pdf and prompt_question:
            st.write("Uploaded PDFs and CSVs")
            documents_combine=csv_processing(uploaded_files_csv)
            documents_combine.extend(pdf_processing(uploaded_files_pdf))
            prompt_retrieval(documents_combine,prompt_question,llm,prompt)
    else:
            st.write("Files not uploaded")
            pass
        

