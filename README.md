# 🩺 **Generative AI Medical Report Analyzer**

## Overview
Welcome to the **Generative AI Medical Report Analyzer**! This project leverages **Generative AI**, **LLMs (Large Language Models)**, and **RAG (Retrieval-Augmented Generation)** to process medical reports in PDF format and provide comprehensive analysis, health recommendations, and personalized advice.

The tool is designed to help individuals and healthcare professionals interpret medical reports, making them more accessible and understandable for a general audience. It combines **AI-powered insights** with real-time **chatbot interactions** to guide users through the analysis.

---

## ⚙️ **Features**
- 📄 **Upload and Analyze PDF Reports**: Easily upload your medical reports in PDF format for analysis.
- 🧠 **Generative AI Analysis**: The system uses advanced **Generative AI** to extract summaries and provide health recommendations based on the medical content.
- 🤖 **Interactive Chatbot**: Ask the chatbot questions related to your medical report, and it will provide tailored responses using
AI-powered insights.
- 🏋️‍♂️ **Wellness Mode**: Input personal health data (age, weight, lifestyle) to receive personalized wellness advice.
- 📈 **Health Insights**: View detailed insights into potential health risks, recommended lifestyle changes, and much more.

---

## 🔧 **Tech Stack**
- **Backend**: Built with **FastAPI** for efficient handling of requests and processing.
- **AI Models**: Uses GPT-4o-mini for natural language understanding and content generation.
- **RAG (Retrieval-Augmented Generation)**: Integrates data from medical reports , uses OPENAI text small embeddings and external research sources to provide enhanced insights.
- **Frontend**: Beautiful, responsive UI powered by **HTML**, **CSS**, and **JavaScript**.

---

## 🛠 **How It Works**

1. **Upload your medical report**: The system accepts medical reports in PDF format.
2. **Generative AI analyzes the report**: The AI extracts key information and generates a **summary** and **health recommendations**.
3. **Interactive Chatbot**: Use the chatbot to ask questions about your report and receive detailed, context-aware responses.
4. **Wellness Mode**: Enter your personal details (age, weight, gender, and lifestyle) to get **personalized wellness advice**.

---

## 📄 **Getting Started**

To get started with the **Generative AI Medical Report Analyzer**, follow these steps:

1. **Install Dependencies**:

    Ensure you have **Python 3.10** installed, then run:

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the Application**:

    To start the backend server, run:

    ```bash
    uvicorn main:app --reload
    ```

3. Open the application in your browser at `http://127.0.0.1:8000`.

---

## 📸 **Screenshots**
Here are some screenshots of the application in action:

*Upload your medical Report:-*
![Screenshot 1](images/Screenshot%202024-12-14%20052642.png)

*Report Analysis:-*
![Screenshot 2](images/Screenshot%202024-12-14%20052711.png)

*Health Recommendations:-*
![Screenshot 3](images/Screenshot%202024-12-14%20052749.png)

*User Chatbot Query:-*
![Screenshot 4](images/Screenshot%202024-12-14%20052814.png)

*Chatbot response:-*
![Screenshot 5](images/Screenshot%202024-12-14%20052827.png)

*Wellness Mode(For general users):-*
![Screenshot 6](images/Screenshot%202024-12-14%20052855.png)

*Wellness Response 1:-*
![Screenshot 7](images/Screenshot%202024-12-14%20052920.png)

*Wellness Response 2:-*
![Screenshot 8](images/Screenshot%202024-12-14%20052935.png)


---

## 💡 **Future Directions**
- 🌍 **Multi-Language Support**: Adding translations for different languages to broaden accessibility.
- 📊 **Predictive Analytics**: Predict disease progression and complications based on health data.
- 🧬 **Symptom Checker Integration**: Allow users to input symptoms alongside reports for better diagnostic correlation.
- 💬 **Voice-based Interaction**: Implement voice recognition for better accessibility, especially for those with disabilities.
- 🔒 **Enhanced Security**: Improve data privacy and security features, ensuring that user data is protected.

---

## 🌱 **Support and Acknowledgments**
This project was developed with the goal of improving healthcare accessibility through the power of **Generative AI**, **RAG**, and **LLMs**. Special thanks to the contributors, open-source community, and various research papers that made this possible.

