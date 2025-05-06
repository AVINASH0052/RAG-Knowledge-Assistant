# app.py
import os
import re
import operator
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
    """Initialize NVIDIA client with secrets management"""
    if "API_KEY" not in st.secrets:
        raise ValueError("NVIDIA_API_KEY not found in Streamlit secrets")
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=st.secrets["API_KEY"]
    )

nvidia_client = init_nvidia_client()

@st.cache_resource(show_spinner="Loading embeddings...")
def get_embeddings():
    """Cached embeddings model loader"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

def load_and_chunk_documents():
    """Document processor with validation"""
    if not os.path.exists(DOCUMENT_DIR):
        raise FileNotFoundError(f"Missing document directory: {DOCUMENT_DIR}")

    documents = []
    for file in ["doc1.txt", "doc2.txt", "doc3.txt"]:
        file_path = os.path.join(DOCUMENT_DIR, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing document: {file_path}")
        documents.extend(TextLoader(file_path).load())

    return RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    ).split_documents(documents)

@st.cache_resource(show_spinner="Initializing vector store...")
def initialize_vector_store():
    """Vector store manager with caching"""
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
    """Safe answer generation with sanitization"""
    system_prompt = f"""Answer the question using ONLY this context.
    Be concise and professional. If unsure, state "I don't know".
    
    Context: {context}"""
    
    try:
        response = nvidia_client.chat.completions.create(
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
    """Response sanitizer"""
    return text.replace("<think>", "").replace("</think>", "").strip()

def math_calculator(query):
    """Secure calculation handler"""
    try:
        allowed_ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv
        }
        
        # Enhanced pattern to handle spaces and different number formats
        match = re.search(r'(\d+\.?\d*)\s*([\+\-\*\/])\s*(\d+\.?\d*)', query)
        if not match:
            return "❌ Invalid format. Use: 'number operator number' (e.g., 15 * 3)"
            
        num1, op, num2 = match.groups()
        result = allowed_ops[op](float(num1), float(num2))
        
        # Formatting for whole numbers vs decimals
        return str(int(result)) if result.is_integer() else f"{result:.2f}"
        
    except ZeroDivisionError:
        return "❌ Cannot divide by zero"
    except Exception as e:
        return f"❌ Calculation error: {str(e)}"

def route_query(query):
    """Enhanced query router"""
    query = query.lower()
    # More precise calculation detection
    if re.search(r"\b(calculate|compute|solve|what is|convert|math)\b", query):
        return "calculator", math_calculator(query)
    if re.search(r"\b(define|explain|meaning of)\b", query):
        return "dictionary", term_definition(query)
    return "rag", None

def term_definition(query):
    """Technical term resolver"""
    tech_terms = {
        "OLED": "Organic Light-Emitting Diode display technology",
        "IP68": "Ingress Protection rating for dust/water resistance",
        "5G": "Fifth-generation cellular network technology"
    }
    match = re.search(r"\b(define|what is|meaning of) (\w+)", query, re.IGNORECASE)
    return tech_terms.get(match.group(2).upper(), "❌ Term not found") if match else "❌ No term specified"

def route_query(query):
    """Enhanced query router"""
    query = query.lower()
    if re.search(r"\b(calculate|compute|math)\b|[\+\-\*\/]", query):
        return "calculator", math_calculator(query)
    if re.search(r"\b(define|what is|meaning of)\b", query):
        return "dictionary", term_definition(query)
    return "rag", None

def main():
    """Streamlit application interface"""
    st.title("🧠 RAG-Powered Multi-Agent Q&A Assistant")
    
    try:
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
        with st.spinner("Analyzing documents..."):
            context_docs = vs.similarity_search(query, k=3)
            context = "\n---\n".join([doc.page_content for doc in context_docs])
            answer = generate_answer(query, context)

    st.subheader("Analysis")
    col1, col2 = st.columns(2)
    col1.metric("Processing Path", tool.upper())
    col2.metric("Context Sources", len(context_docs) if context else 0)
    
    if context:
        with st.expander("View Relevant Context"):
            st.text(context)
    
    st.subheader("Answer")
    st.markdown(f"```\n{answer}\n```")

if __name__ == "__main__":
    main()
