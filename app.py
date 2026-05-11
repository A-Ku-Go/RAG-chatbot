import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
import os
import chardet
import logging

# --- Suppress Streamlit's internal warnings at the very top ---
streamlit_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith('streamlit')]
for logger in streamlit_loggers:
    logger.setLevel(logging.ERROR) # Set to ERROR to hide WARNING, INFO, DEBUG

# --- Helper Functions (with cache_key_trigger for explicit invalidation) ---

# REMOVED THE UNDERSCORE from 'documents' argument
@st.cache_resource
def get_text_chunks(_documents, cache_key_trigger): # cache_key_trigger forces re-run
    """Splits loaded documents into smaller, manageable text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Max characters per chunk
        chunk_overlap=200, # Overlap between chunks to maintain context
        length_function=len # Uses character count for chunking
    )
    chunks = text_splitter.split_documents(_documents)
    return chunks

# REMOVED THE UNDERSCORE from 'text_chunks' argument
@st.cache_resource
def get_vector_store(_text_chunks, cache_key_trigger): # cache_key_trigger forces re-run
    """Creates and persists a vector store from text chunks using embeddings."""
    # Using 'all-MiniLM-L6-v2' for efficient sentence embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(_text_chunks, embeddings)
    return vector_store

# REMOVED THE UNDERSCORE from 'vector_store' argument
@st.cache_resource
def get_conversation_chain(_vector_store, cache_key_trigger): # cache_key_trigger forces re-run
    """Initializes and returns a conversational retrieval chain with Gemini LLM."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", # Using the fast Gemini Flash model
        temperature=0.7, # Controls creativity (0.0-1.0)
        google_api_key=st.secrets["GOOGLE_API_KEY"] # Securely retrieves API key from secrets.toml
    )
    # Conversation buffer memory to store chat history
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    # ConversationalRetrievalChain combines LLM with retriever for Q&A over documents
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vector_store.as_retriever(), # Retriever to fetch relevant documents
        memory=memory
    )
    return conversation_chain

def load_document(file):
    """Loads a single document based on its file type."""
    file_extension = os.path.splitext(file.name)[1].lower()
    
    # Create a temporary file path
    temp_file_path = os.path.join("./temp_files", f"{os.urandom(8).hex()}_{file.name}")
    os.makedirs("./temp_files", exist_ok=True) # Ensure temp_files directory exists

    # Write the uploaded file's content to the temporary file
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())

    docs = None
    loader_map = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
    }

    loader_class = loader_map.get(file_extension)

    if loader_class is None:
        st.error(f"Unsupported file type: {file_extension}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path) # Clean up temp file
        return None

    try:
        # --- MODIFICATION START for .txt encoding ---
        if file_extension == ".txt":
            # Attempt to automatically detect encoding for text files
            # This requires 'chardet' library
            loader = TextLoader(temp_file_path, autodetect_encoding=True)
        else:
            loader = loader_class(temp_file_path)
        # --- MODIFICATION END ---
        
        docs = loader.load()
    except Exception as e:
        st.error(f"Error loading {file.name}: {e}")
    finally:
        # Ensure temporary file is removed after processing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return docs

# --- Streamlit UI and Main Logic ---

def main():
    st.set_page_config(page_title="Chat with Multiple Documents", page_icon="📚")
    st.title("📚 Retrieval-Augmented Generation (RAG) App Using Streamlit")

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = False
    # Unique identifier for document sets to force cache invalidation
    if "document_set_id" not in st.session_state:
        st.session_state.document_set_id = 0 # Start with 0

    # Initial guidance for the user
    if not st.session_state.processed_documents:
        st.info("Upload your documents (PDF, DOCX, TXT) in the sidebar and click 'Process Documents' to start chatting!")

    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("Your documents")
        uploaded_files = st.file_uploader(
            "Upload your PDF, DOCX, or TXT files here and click 'Process'",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"],
            key="file_uploader" # Unique key for the uploader widget
        )
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                if uploaded_files:
                    all_documents = []
                    for uploaded_file in uploaded_files:
                        with st.spinner(f"Loading {uploaded_file.name}..."):
                            docs = load_document(uploaded_file)
                            if docs:
                                all_documents.extend(docs)

                    if all_documents:
                        # Increment document_set_id to force cache invalidation for new document sets
                        st.session_state.document_set_id += 1 

                        with st.spinner("Splitting text into chunks..."):
                            # Pass the new document_set_id as a cache key trigger
                            text_chunks = get_text_chunks(all_documents, st.session_state.document_set_id)
                        
                        with st.spinner("Creating vector store and embeddings..."):
                            # Pass the new document_set_id as a cache key trigger
                            vector_store = get_vector_store(text_chunks, st.session_state.document_set_id)
                        
                        # Pass the new document_set_id as a cache key trigger for conversation chain
                        st.session_state.conversation = get_conversation_chain(vector_store, st.session_state.document_set_id)
                        
                        st.session_state.processed_documents = True
                        st.success("Documents processed successfully! You can now ask questions.")
                        st.session_state.chat_history = [] # Clear chat history for new docs
                    else:
                        st.warning("No documents were loaded or processed successfully. Please upload valid files.")
                        st.session_state.processed_documents = False
                else:
                    st.warning("Please upload at least one document.")
                    st.session_state.processed_documents = False

    # Main chat interface
    st.header("Ask a question about your documents:")
    user_question = st.chat_input("Your question:")

    if user_question:
        if st.session_state.processed_documents and st.session_state.conversation:
            with st.spinner("Generating response..."):
                # The conversation chain expects a dictionary with 'question'
                response = st.session_state.conversation({'question': user_question})
                
                # Append user question and assistant response to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
        else:
            st.warning("Please upload and process documents first to start chatting.")

    # Display chat messages in chronological order
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Entry point for the Streamlit application
if __name__ == '__main__':
    main()