========================================================================
DOCUMENT QUESTION WITH RAG (RETRIEVAL-AUGMENTED GENERATION)
========================================================================

1. OVERVIEW
------------------------------------------------------------------------
This application is a Streamlit-based Document QA System that allows 
users to upload PDF files and ask questions strictly based on the 
uploaded document's content.

Key Features:
- PDF Text Extraction using `pdfplumber` (handles standard digital PDFs).
- Chunking & Vectorization via LangChain and Chroma DB.
- Embeddings using OpenAI's `text-embedding-3-small`.
- Text Generation using OpenAI's `gpt-4o-mini`.
- Metadata-tagged document overview/summary generation upon upload.
- Distance threshold filtering to prevent irrelevant context retrieval.
- Strict anti-hallucination prompt engineering.


2. PREREQUISITES & DEPENDENCIES
------------------------------------------------------------------------
Make sure Python 3.9+ is installed. 

Required Python Packages:
  - streamlit
  - langchain
  - langchain-openai
  - langchain-chroma
  - langchain-community
  - langchain-text-splitters
  - pdfplumber
  - python-dotenv

To install all dependencies, run:
  pip install streamlit langchain langchain-openai langchain-chroma langchain-community langchain-text-splitters pdfplumber python-dotenv


3. ENVIRONMENT SETUP
------------------------------------------------------------------------
1. Create a file named `.env` in the root folder of the project.
2. Add your OpenAI API key inside the `.env` file:

   OPENAI_API_KEY=your_actual_openai_api_key_here

3. Ensure the `.env` file is in the same directory as your Python script.


4. HOW TO RUN THE APPLICATION
------------------------------------------------------------------------
Run the application using Streamlit from your terminal/command prompt:

  streamlit run app.py

(Replace `app.py` with the filename of your main Python script if different).


5. OPERATING INSTRUCTIONS (STEP-BY-STEP)
------------------------------------------------------------------------
STEP 1: Upload a PDF
  - Use the sidebar upload box ("Upload PDF") to select your file.

STEP 2: Process the PDF
  - Click "Process and Add to Database".
  - The system extracts text using `pdfplumber`, splits the text into 
    chunks (1000 characters, 200 overlap), and generates an automated 
    overview document tagged with metadata `type: overview`.
  - Wait until the success message appears. The chat input will unlock 
    automatically.

STEP 3: Asking Questions
  - Type questions into the chat bar at the bottom.
  - Ask for general summaries using queries like: "summary of file", 
    "what is inside", or "tell me about this document".
  - Ask specific question queries strictly related to text content.

STEP 4: Debugging & Inspection
  - Expand the "🛠️ Debug Info & Retrieved Documents" panel below any 
    response to inspect the raw similarity distance scores and verify 
    which exact text chunks were retrieved for the LLM.


6. UNDERSTANDING THE SYSTEM SETTINGS & RULES
------------------------------------------------------------------------
A. Relevance Threshold (Max Distance)
   - Chroma uses distance metrics (L2 / Cosine). For OpenAI 
     embeddings (`text-embedding-3-small`), distance scores typically 
     range between 0.8 and 1.8.
   - Adjust the "Relevance threshold (Max Distance)" slider in the 
     sidebar if search results are missed or if too many irrelevant 
     chunks pass through. The recommended default is 2.0.

B. Strict Document QA Rule
   - The LLM is strictly instructed:
     "If the answer cannot be found directly in the provided context, 
      state: 'I cannot answer this question based on the provided document.'"
   - If a term appears in your document (e.g., as a prerequisite or bullet 
     point) but is not explicitly defined, the model will decline to 
     invent a definition to prevent hallucination.

C. Handling Scanned / Image-Based PDFs
   - The app first extracts embedded digital text with `pdfplumber`.
   - If the PDF is scanned or contains only images, the app automatically
     renders the pages and sends them through Tesseract OCR.
   - On Windows, install the Tesseract OCR engine and add its install folder
     to PATH. The Python wrapper is included in `requirements.txt`.


7. MAINTENANCE & CLEANUP
------------------------------------------------------------------------
- Switching PDFs: Uploading a new PDF automatically resets chat history 
  and requires clicking "Process and Add to Database".
- Cleaning Database: Click "🗑️ Clean Database" in the sidebar to 
  permanently clear the local vector store (`./chroma_db`) and reset 
  the application state.
========================================================================