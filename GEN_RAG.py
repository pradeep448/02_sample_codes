# 1. Install all required packages using pip before running the code:
# pip install langchain openai faiss-cpu pypdf streamlit

# 2. Ingest & preprocess the PDF manual
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the PDF doc
pdf_path = "document.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the document into overlapping textual chunks for better context retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# 3. Create embeddings for each chunk and store them in a FAISS vector database
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Generate embeddings for the text chunks using an OpenAI model
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Set up the Retrieval-Augmented Generation (RAG) chain: Retrieve relevant context and generate answers using an LLM
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Initialize the LLM (e.g., GPT-3.5-turbo)
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Build a chain that retrieves context from FAISS and generates an answer
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# 5. Define a helper function to answer questions
def answer_question(question):
    """Query the RAG pipeline and return an answer based on the document."""
    return qa_chain.run(question)

# 6. Streamlit user interface for Q&A
import streamlit as st

# Set up the app title
st.title("Document Q&A")

# Input box for user questions
user_query = st.text_input("Ask a question about your Document:")

# Display the answer when a question is submitted
if user_query:
    answer = answer_question(user_query)
    st.write("**Answer:**", answer)
