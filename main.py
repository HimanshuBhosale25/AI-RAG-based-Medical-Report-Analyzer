from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
import tempfile
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()

# Function to extract text from PDF using PyPDFLoader
def extract_text_from_pdf(pdf_file: BytesIO):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf_file:
        temp_pdf_file.write(pdf_file.read())
        temp_pdf_path = temp_pdf_file.name

    loader = PyPDFLoader(temp_pdf_path)
    try:
        documents = loader.load()
    finally:
        os.remove(temp_pdf_path)  # Ensure temporary file is deleted
    return documents

# New helper function to generate general health recommendations based on the summary
def generate_general_health_recommendations(summary):
    # Define the prompt template for LangChain
    prompt = """
    Based on the following medical report summary, provide general health recommendations for the patient.Consider you are responsding to the patient instead of general response . The summary is: {summary}.
    Focus on general lifestyle advice, medication adherence, and preventive care.It should be short and concise.
    """
    
    # Create the prompt with the specific summary
    template = PromptTemplate(input_variables=["summary"], template=prompt)
    
    # Initialize LangChain's OpenAI LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Set up LangChain's LLMChain to process the prompt
    chain = LLMChain(llm=llm, prompt=template)
    
    # Run the chain to generate health recommendations
    health_recommendations = chain.run({"summary": summary})
    
    return health_recommendations

# Main function to run the RAG workflow
def analyze_medical_report(pdf_file: BytesIO):
    documents = extract_text_from_pdf(pdf_file)
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-small"
    )
    faiss_index = FAISS.from_documents(documents, embeddings)
    retriever = faiss_index.as_retriever()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = chain.invoke({"query": "Summarize the medical report. The summary should be so simple that a person with non-medical background can also understand it. Also explain the complex medical terms.You have to respond as if your talking to the patient.It should be short and concise."})
    
    summary = response['result']

    # Generate general health recommendations based on the summary using LangChain
    health_recommendations = generate_general_health_recommendations(summary)

    return {
        "summary": summary,
        "health_recommendations": health_recommendations
    }
