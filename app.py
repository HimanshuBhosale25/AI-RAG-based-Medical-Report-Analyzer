from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from io import BytesIO
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel
from main import analyze_medical_report  # Import your analyze_medical_report function

# Initialize FastAPI app
app = FastAPI()

# Serve the HTML frontend
app.mount("/static", StaticFiles(directory="templates"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("templates/index.html")

@app.post("/analyze-report/")
async def upload_and_analyze(pdf: UploadFile = File(...)):
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        pdf_file = BytesIO(await pdf.read())
        result = analyze_medical_report(pdf_file)
        return {"summary": result["summary"], "health_recommendations": result["health_recommendations"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Define Pydantic model for the wellness request
class WellnessRequest(BaseModel):
    age: int
    weight: int
    lifestyle: str
    gender: str 

@app.post("/wellness-mode/")
async def wellness_mode(request: WellnessRequest):
    wellness_advice = generate_wellness_advice(request.age, request.weight, request.lifestyle, request.gender)
    return {"wellness_advice": wellness_advice}

def generate_wellness_advice(age: int, weight: int, lifestyle: str, gender: str) -> str:
    # Define the prompt template for LangChain
    prompt = """
    You are a wellness expert. Provide personalized wellness advice based on the following information:
    Age: {age}
    Weight: {weight} kg
    Gender: {gender}  # Include gender in the prompt
    Lifestyle: {lifestyle} (choose from sedentary, active, very_active)

    Provide advice on:
    - Diet recommendations
    - Physical activity
    - Preventive care based on age, gender, and lifestyle
    """
    
    # Create the prompt with the specific variables
    template = PromptTemplate(input_variables=["age", "weight", "lifestyle", "gender"], template=prompt)

    # Initialize the OpenAI LLM using LangChain (GPT-4 Mini)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    # Set up LangChain's LLMChain to process the prompt
    chain = LLMChain(llm=llm, prompt=template)
    
    # Get wellness advice from the LangChain model
    wellness_advice = chain.run({"age": age, "weight": weight, "lifestyle": lifestyle, "gender": gender})
    
    return wellness_advice

# Define Pydantic model for the chatbot interaction
class ChatbotRequest(BaseModel):
    user_message: str
    context: str

@app.post("/chatbot/")
async def chatbot_interaction(request: ChatbotRequest):
    user_message = request.user_message
    context = request.context
    try:
        # Initialize the OpenAI LLM for Chatbot interaction (GPT-4 Mini)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        # Construct the prompt, including the summary as context
        prompt = f"Context: {context}\nUser: {user_message}\nBot:"
        
        # Correct way to call the LLM using LangChain
        response = llm.predict(prompt)  # Use predict() instead of just calling the model directly
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chatbot request: {str(e)}")