import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import tempfile

# Load environment variables first
load_dotenv()

# Move API key to environment variable for better security
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file.seek(0)
            pdf_reader = PdfReader(tmp_file.name)
            for page in pdf_reader.pages:
                text += page.extract_text()
        os.unlink(tmp_file.name)
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
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Save with allow_dangerous_deserialization enabled
        vector_store.save_local("faiss_index", allow_dangerous_deserialization=True)
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
    model = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.3,
        api_key=GROQ_API_KEY
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        # Load with allow_dangerous_deserialization enabled
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
        st.error(f"An error occurred while processing your question: {str(e)}")

def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖")

    with st.sidebar:
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files & Click on the Submit & Process Button",
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
        st.write("AI App created by @ Bagyalakshmi Shinde")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")
    if user_question:
        if not os.path.exists("faiss_index"):
            st.warning("Please upload and process PDF files before asking questions.")
            return
        user_input(user_question)

    st.markdown(
        """
        © Bagyalakshmi S Shinde| Made with ❤️
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
