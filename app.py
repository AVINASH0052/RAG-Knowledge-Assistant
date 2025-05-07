# app.py 
import os
import re
import numexpr
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
DOCUMENT_DIR = "docs"
VECTOR_STORE_NAME = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=False)
def init_nvidia_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=st.secrets["API_KEY"]
    )

@st.cache_resource(show_spinner="🚀 Loading embeddings...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu', 'token': st.secrets["HF_TOKEN"]},
        encode_kwargs={'normalize_embeddings': False}
    )

def load_and_chunk_documents():
    if not os.path.exists(DOCUMENT_DIR):
        raise FileNotFoundError(f"Missing document directory: {DOCUMENT_DIR}")
    
    documents = []
    for file in ["doc1.txt", "doc2.txt", "doc3.txt"]:
        file_path = os.path.join(DOCUMENT_DIR, file)
        if os.path.exists(file_path):
            documents.extend(TextLoader(file_path).load())
    
    return RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    ).split_documents(documents)

@st.cache_resource(show_spinner="📚 Building knowledge base...")
def initialize_vector_store():
    if os.path.exists(VECTOR_STORE_NAME):
        return FAISS.load_local(
            VECTOR_STORE_NAME,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )
    
    chunks = load_and_chunk_documents()
    vector_store = FAISS.from_documents(chunks, get_embeddings())
    vector_store.save_local(VECTOR_STORE_NAME)
    return vector_store

def generate_answer(query, context):
    system_prompt = f"""Answer using ONLY this context. Be concise.
    If unsure, say "I don't know".
    
    Context: {context}"""
    
    try:
        response = st.session_state.nvidia_client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.6,
            max_tokens=1024
        )
        return clean_response(response.choices[0].message.content)
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def clean_response(text):
    return text.replace("<think>", "").replace("</think>", "").strip()

def math_calculator(query):
    """Enhanced math processor with natural language support"""
    try:
        # Normalize textual operators
        normalized = query.lower()
        normalized = re.sub(r'\bplus\b', '+', normalized)
        normalized = re.sub(r'\bminus\b', '-', normalized)
        normalized = re.sub(r'\btimes\b', '*', normalized)
        normalized = re.sub(r'\bdivided by\b', '/', normalized)
        
        # Remove non-math characters
        clean_query = re.sub(r'[^0-9\.\+\-\*/]', '', normalized)
        
        if not clean_query:
            return "❌ Invalid math expression"
        
        # Validate expression
        if not re.fullmatch(r'^[\d\.\+\-\*/]+$', clean_query):
            return "❌ Invalid math expression"
        
        # Evaluate using numexpr for safety
        result = numexpr.evaluate(clean_query).item()
        return f"{result:.2f}" if isinstance(result, float) else str(int(result))
    
    except ZeroDivisionError:
        return "❌ Cannot divide by zero"
    except Exception as e:
        return f"❌ Calculation error: {str(e)}"

def term_definition(query):
    tech_terms = {
        "OLED": "Organic Light-Emitting Diode display technology",
        "IP68": "Ingress Protection rating for dust/water resistance",
        "5G": "Fifth-generation cellular network technology"
    }
    match = re.search(r"\b(define|what is|meaning of) (\w+)", query, re.IGNORECASE)
    return tech_terms.get(match.group(2).upper(), "❌ Term not found") if match else "❌ No term specified"

def route_query(query):
    """Improved query routing with enhanced math detection"""
    # Enhanced math pattern detection
    math_pattern = r'(?i)(\d+\s*(plus|minus|times|divided by|\+|\-|\*|/)\s*\d+)'
    if re.search(math_pattern, query):
        return "calculator", math_calculator(query)
    
    # Technical definitions
    if re.search(r'\b(define|what is|meaning of)\b', query):
        return "dictionary", term_definition(query)
    
    # Default to RAG
    return "rag", None

def main():
    st.title("🧠 RAG-Powered Multi-Agent Q&A Assistant")
    
    try:
        if 'nvidia_client' not in st.session_state:
            st.session_state.nvidia_client = init_nvidia_client()
            
        vs = initialize_vector_store()
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return

    query = st.text_input("Ask your question:", placeholder="Type your question here...")
    if not query.strip():
        return

    tool, answer = route_query(query)
    context = None

    if tool == "rag":
        with st.spinner("🔍 Analyzing documents..."):
            context_docs = vs.similarity_search(query, k=3)
            context = "\n---\n".join([doc.page_content for doc in context_docs])
            answer = generate_answer(query, context)

    st.subheader("Analysis")
    cols = st.columns(2)
    cols[0].metric("Processing Path", tool.upper())
    cols[1].metric("Context Sources", len(context_docs) if context else 0)
    
    if context:
        with st.expander("📖 View Relevant Context"):
            st.text(context)
    
    st.subheader("Answer")
    st.markdown(f"```\n{answer}\n```")

if __name__ == "__main__":
    main()
