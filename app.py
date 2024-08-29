# Import libraries
import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BertConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import re
from langchain.chains.question_answering import load_qa_chain
import PyPDF2

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()

    text    = re.sub(r'\n', ' ', text)        
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
   
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def preprocess_pdf(pdf_file):
    """Extracts text and cleans the uploaded PDF."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # Clean text (remove headers, footers, etc.)
    return text

def answer_question(question, model, tokenizer, max_length=512): # Increased max_length
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    """Generates an answer based on question, PDF text, and LLM model."""
    
    prompt_template = f"""Answer the question as detailed as possible from the 
    provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, answer is not
    available in the context, don't provide the wrong answer
    \n\nContext:\n {docs}?\n Question: \n{question}\n 
    Answer:return [ document ( page _ content = as a answer from context ) ]"""

   
    inputs = tokenizer(prompt_template, return_tensors="pt", truncation=True, max_length=max_length) # Added truncation and max_length
   
    available_tokens = max(1, max_length - inputs['input_ids'].shape[1])
    output = model.generate(**inputs, max_new_tokens=available_tokens,do_sample=True, temperature=0.01) 
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
  
    st.write("answer: ", answer)
    st.success("Done")
    

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Hugging face model💁")
    # Load pre-trained Equall/Saul-Instruct-v1 model and tokenizer
    model_name = "nlpaueb/legal-bert-base-uncased"
    max_length=512
    config = BertConfig.from_pretrained(model_name) 

    # Use AutoModelForCausalLM instead of AutoModelForSeq2SeqLM
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config) 
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length) # Increased model_max_length for tokenizer

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        answer_question(user_question, model, tokenizer) # Pass max_length to answer_question

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
  main()