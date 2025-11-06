"""
RAG Chatbot for university notes
Deploy on Streamlit Cloud with Groq API
"""

import os
# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="🎓 Notes Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Import other libraries after set_page_config
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    from groq import Groq
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("Run: pip install chromadb sentence-transformers groq")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load embedding model and ChromaDB"""
    try:
        # Embedding model
        with st.spinner("Loading embedding model..."):
            embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # ChromaDB
        with st.spinner("Loading vector database..."):
            client = chromadb.PersistentClient(path="vectorstore")
            collection = client.get_collection("appunti_corso")
        
        return embedding_model, collection
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure you have run `python preprocess.py` first to create the vectorstore")
        return None, None

def process_uploaded_file(uploaded_file, collection, embedding_model):
    """Process and add uploaded file to vector database"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load document based on file type
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        else:
            os.unlink(tmp_file_path)
            return False, "Unsupported file type. Please upload PDF or TXT files."
        
        # Load and split
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add to ChromaDB
        for i, chunk in enumerate(chunks):
            embedding = embedding_model.encode([chunk.page_content])[0].tolist()
            
            collection.add(
                ids=[f"{uploaded_file.name}_{i}"],
                embeddings=[embedding],
                documents=[chunk.page_content],
                metadatas=[{"source": uploaded_file.name, "chunk": i}]
            )
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return True, f"Successfully added {len(chunks)} chunks from {uploaded_file.name}"
    
    except Exception as e:
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)
        return False, f"Error processing file: {str(e)}"


def retrieve_context(query: str, collection, embedding_model, n_results: int = 3):
    """Retrieve the most relevant chunks"""
    try:
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        contexts = []
        sources = set()
        
        if results['documents'] and len(results['documents']) > 0:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                contexts.append(doc)
                sources.add(metadata.get('source', 'Unknown'))
        
        return "\n\n".join(contexts), list(sources)
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return "", []

def generate_response(query: str, context: str, groq_client, model: str = "llama-3.3-70b-versatile"):
    """Generate response using Groq"""
    try:
        prompt = f"""You are an assistant helping university students study. 
Answer the question based ONLY on the context provided from the course notes.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer in English clearly and concisely
- If the answer is not in the context, say so honestly
- Use examples from the context when possible
- Do not make up information
- If the user greets you, greet back and ask how can you help 

ANSWER:"""

        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an educational assistant helping students study."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Response generation error: {str(e)}"

def main():
    # Header
    st.markdown("<h1 class='main-header'>🎓 Finance & Banking Notes Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Ask questions about your notes using RAG</p>", unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "llama-3.3-70b-versatile"
    
    # Sidebar for API key
    with st.sidebar:
        
        st.markdown("### 👥 Contributors")
        st.markdown("""
        - Lorenzo Brontesi
        - Francesco Ansanelli
        """)
        
        # Try to get API key from secrets first
        groq_api_key = None
        
        if "GROQ_API_KEY" in st.secrets:
            groq_api_key = st.secrets["GROQ_API_KEY"]
            #st.success("✅ API Key loaded from secrets")
        else:
            st.header("⚙️ Configuration")
            # If not in secrets, ask user to input
            st.info("💡 **Tip**: Add your key to `.streamlit/secrets.toml` to avoid typing it every time")
            
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Get your free key at https://console.groq.com",
                value=""
            )
            
            if not groq_api_key or groq_api_key == "":
                st.warning("⚠️ Insert your Groq API Key to start")
                st.info("👉 Sign up for free at [Groq Console](https://console.groq.com)")
                st.markdown("---")
                st.subheader("📝 How to use secrets")
                st.markdown("Create `.streamlit/secrets.toml`:")
                st.code('GROQ_API_KEY = "gsk_your_key_here"', language="toml")
                st.stop()
            else:
                st.success("✅ API Key configured")
            
        st.markdown("---")
        # File Upload Section
        st.header("📤 Upload Documents")
        uploaded_file = st.file_uploader(
            "Upload PDF or TXT file",
            type=['pdf', 'txt'],
            help="Upload course notes to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("➕ Add to Knowledge Base", type="primary"):
                # Load models first
                embedding_model, collection, _ = load_models()
                
                if embedding_model and collection:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        success, message = process_uploaded_file(uploaded_file, collection, embedding_model)
                    
                    if success:
                        st.success(message)
                        # Force reload of models to refresh collection
                        st.cache_resource.clear()
                    else:
                        st.error(message)
                else:
                    st.error("Cannot load models")
        
        st.markdown("---")
        st.markdown("### 📚 Info")
        st.markdown("""
        This chatbot uses:
        - **RAG** to search through notes
        - **Groq** (Llama 3.3 70B) to generate responses
        - **ChromaDB** as vector database
        """)
        
        # Model selector
        st.markdown("---")
        model_choice = st.selectbox(
            "🤖 LLM Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            help="Change model if you encounter errors or want different speeds"
        )
        st.session_state.model_choice = model_choice
        
        if st.button("🗑️ Clear history"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize Groq client
    try:
        groq_client = Groq(api_key=groq_api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        st.stop()
    
    # Load models
    embedding_model, collection = load_models()
    
    if embedding_model is None or collection is None:
        st.error("❌ Cannot load models. Make sure you have:")
        st.markdown("""
        1. Run `python preprocess.py` to create the vectorstore
        2. The `vectorstore/` folder exists in your project directory
        3. There are documents in the vectorstore
        """)
        st.stop()
    
    # Show existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📄 Sources"):
                    for source in message["sources"]:
                        st.text(f"• {source}")
    
    # User input
    if prompt := st.chat_input("Ask a question about your notes..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching through notes..."):
                # Retrieve
                context, sources = retrieve_context(prompt, collection, embedding_model)
                
                if not context:
                    response = "I couldn't find relevant information in the notes to answer this question."
                    sources = []
                else:
                    # Generate
                    with st.spinner("💭 Generating response..."):
                        response = generate_response(prompt, context, groq_client, st.session_state.model_choice)
            
            st.markdown(response)
            
            if sources:
                with st.expander("📄 Sources"):
                    for source in sources:
                        st.text(f"• {source}")
        
        # Save response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Critical error: {e}")
        st.exception(e)