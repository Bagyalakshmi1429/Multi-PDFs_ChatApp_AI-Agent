import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50000, 
        chunk_overlap=1000,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        # Save without the allow_dangerous_deserialization parameter
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Only use allow_dangerous_deserialization when loading
        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")
    
    if user_question:
        if not os.path.exists("faiss_index"):
            st.warning("Please upload and process PDF files before asking questions.")
            return
        user_input(user_question)

    with st.sidebar:
        try:
            st.image("img/Robot.jpg")
        except:
            st.write("Robot image not found")
        st.write("---")
        
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files & \n Click on the Submit & Process Button ",
            accept_multiple_files=True
        )
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
                
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.warning("No text could be extracted from the uploaded PDFs.")
                        return
                        
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    if vector_store is not None:
                        st.success("Processing complete! You can now ask questions about your documents.")
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
        
        st.write("---")
        # try:
        #     st.image("img/gkj.jpg")
        # except:
        #     st.write("Author image not found")
        st.write("AI App created by @ Bagyalakshmi S Shinde")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/Bagyalakshmi1429/Multi-PDFs_ChatApp_AI-Agent" target="_blank">Bagyalakshmi S Shinde</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
