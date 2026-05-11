# Document Chat RAG App

A Streamlit app that lets you upload PDF, DOCX, or TXT documents and ask questions using a Retrieval-Augmented Generation (RAG) workflow.

## Features

- Upload multiple documents at once
- Supports PDF, DOCX, and TXT files
- Splits documents into text chunks for retrieval
- Uses sentence-transformer embeddings with FAISS vector search
- Connects to Google Gemini via `langchain-google-genai`
- Maintains chat history in Streamlit session state

## Requirements

- Python 3.10+ recommended
- `venv` or another virtual environment

## Installation

1. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Configuration

The app uses a Google API key to connect to Gemini. Store it in Streamlit secrets, for example in `./.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_google_api_key_here"
```

## Usage

Run the Streamlit app from the project root:

```powershell
streamlit run app.py
```

Then open the local URL displayed by Streamlit in your browser.

## How it works

1. Upload files via the sidebar.
2. Click `Process Documents` to load and process files.
3. The app converts each document to text, splits it into chunks, and creates a FAISS vector store.
4. Ask questions in the chat interface.
5. The app retrieves relevant chunks and generates answers using Gemini.

## Supported file types

- `.pdf`
- `.docx`
- `.txt`

## Notes

- Temporary uploaded files are written to `temp_files/` during processing and removed after use.
- If a text file has unknown encoding, the app attempts automatic detection with `chardet`.
- The current implementation uses `gemini-2.0-flash` as the LLM.

## File structure

- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies
- `temp_files/` - temporary upload storage
- `sample files/` - sample or example files (optional)

## Troubleshooting

- If uploads fail, ensure the file type is supported and not corrupted.
- If the app cannot connect to Gemini, verify the Google API key in Streamlit secrets.
- If performance is slow, try smaller documents or increase available RAM for FAISS.
