# Financial Report Question Answering using Retrieval-Augmented Generation (RAG)

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to answer factual financial questions using company annual reports (BMW, Ford, and Tesla).  
The system extracts text from PDFs, embeds the text into a Chroma vector database, and uses an OpenAI language model to retrieve and answer user queries strictly based on the report content.

---

## Project Structure
LLM_NLP/
│
├── data_loader.py # Loads and splits PDF text into manageable chunks
├── rag_pipeline.py # Builds vector embeddings, retriever, and QA chain
├── main.py # Orchestrates data loading, RAG building, and Q&A interface
│
├── Data/
│ ├── BMW/ # BMW Annual Reports (PDF)
│ ├── Ford/ # Ford Annual Reports (PDF)
│ └── Tesla/ # Tesla Annual Reports (PDF)
│
├── requirements.txt # Python dependencies
└── README.md


---

## Setup and Installation

### 1. Clone or unzip the project directory
```bash
cd LLM_NLP
```

---
### 2. Create a Virutal Environment 
python3 -m venv venv
source venv/bin/activate


---
### 3. Install Dependencies 
pip install -r requirements.txt


___
### 4. Set up OpenAI Key
export OPENAI_API_KEY="your_api_key_here"



___
## How to run
python3 main.py